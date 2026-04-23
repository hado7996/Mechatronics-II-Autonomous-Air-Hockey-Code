// test_motor.ino
// -------------------------------------------------------
// H-bot air hockey paddle controller — X axis only.
//
// Architecture:
//   Python owns PID and sends a signed PWM command (-255..255).
//   Arduino applies a kick/spike on start from rest only,
//   then transitions to the commanded PWM.
//
//   For commands below DIAGONAL_THRESHOLD, only motor A runs
//   (at MIN_SPEED PWM) and motor B coasts. This produces
//   diagonal carriage motion, whose X-component is lower
//   than either motor's individual stall floor would allow
//   in a pure-X drive. Y drift is accepted as the tradeoff.
//
// Serial protocol:
//   Receive: "<signed int>\n"   e.g. "127\n" or "-80\n"
//   Send:    nothing (watchdog resets on any received message)
//
// Tuning parameters are all at the top of this file.
// -------------------------------------------------------

#include <avr/io.h>
#include <avr/interrupt.h>

// ── Pin assignments ──────────────────────────────────────
#define MOTOR_A_SV   5
#define MOTOR_A_EN   27
#define MOTOR_A_FR   30
#define MOTOR_A_BK   23
#define MOTOR_A_PG   2
#define MOTOR_A_ALM  29

#define MOTOR_B_SV   9
#define MOTOR_B_EN   40
#define MOTOR_B_FR   31
#define MOTOR_B_BK   24
#define MOTOR_B_PG   3
#define MOTOR_B_ALM  22

// ── Motion limits ────────────────────────────────────────
#define MIN_SPEED          15     // PWM floor — motor stall speed
#define MAX_SPEED          255    // PWM ceiling

// ── Watchdog ─────────────────────────────────────────────
// If no message received for this many ms, brake and wait.
#define WATCHDOG_MS        200

// ── Deadband ─────────────────────────────────────────────
// Commanded PWM values with abs() below this are treated as zero.
// Keeps the motor from buzzing at rest from small Python PID residuals.
#define PWM_DEADBAND       0

// ── Diagonal mode ────────────────────────────────────────
// Commands with abs(pwm) below this threshold drive only motor A
// at MIN_SPEED, producing diagonal carriage motion. The X-component
// is lower than a pure-X drive at MIN_SPEED would produce.
// Set equal to MIN_SPEED or lower to disable diagonal mode.
#define DIAGONAL_THRESHOLD 15

// ── Kick / spike tuning ──────────────────────────────────
//
// KICK_PWM        — PWM applied during the spike (0..255)
// KICK_MAX_MS     — maximum spike duration in milliseconds.
//                   Spike always ends at this timeout even if
//                   target velocity is not reached.
// KICK_TARGET_PPS — if encoder pulses/sec reaches this value
//                   during the spike, end early and hand off.
//                   Set to 0 to disable early exit (pure timeout).
// KICK_MEASURE_US — velocity measurement window in microseconds.
//                   Shorter = more reactive but noisier.
//
#define KICK_PWM           120
#define KICK_MAX_MS        10
#define KICK_TARGET_PPS    80
#define KICK_MEASURE_US    20000

// ── Transition ramp ──────────────────────────────────────
// After the kick ends, ramp from KICK_PWM to the commanded
// PWM over this many milliseconds.
// Set to 0 to disable ramping (hard handoff).
#define KICK_RAMP_MS       10

// ────────────────────────────────────────────────────────
// Internal state — do not tune below this line
// ────────────────────────────────────────────────────────

// ── Encoder ──────────────────────────────────────────────
volatile unsigned long pulseCount    = 0;
static   int           lastCommandDir = 0;

// ── Kick state ───────────────────────────────────────────
static bool          kicking          = false;
static bool          ramping          = false;
static unsigned long kickStartMs      = 0;
static unsigned long rampStartMs      = 0;
static int           rampStartPWM     = 0;
static unsigned long kickWindowStart  = 0;
static unsigned long kickWindowCount  = 0;

// ── Serial parser ─────────────────────────────────────────
static char    serialBuf[16];
static uint8_t serialIdx   = 0;
static int     commandedPWM = 0;       // signed PWM from Python
static unsigned long lastMsgTime = 0;

// ── ISR: Motor A encoder pulse ───────────────────────────
void onPulseA() {
  pulseCount++;
}

// ── Serial poll ──────────────────────────────────────────
void pollSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      serialBuf[serialIdx] = '\0';
      if (serialIdx > 0) {
        commandedPWM = atoi(serialBuf);
        lastMsgTime  = millis();
      }
      serialIdx = 0;
    } else if (c != '\r') {
      if (serialIdx < sizeof(serialBuf) - 1) {
        serialBuf[serialIdx++] = c;
      } else {
        serialIdx = 0;
        memset(serialBuf, 0, sizeof(serialBuf));
      }
    }
  }
}

// ── Atomic direction write ───────────────────────────────
static inline void setBothDirections(bool aForward, bool bForward) {
  uint8_t newBits = 0;
  if (aForward) newBits |= _BV(7);
  if (bForward) newBits |= _BV(6);
  uint8_t oldSREG = SREG;
  cli();
  PORTC = (PORTC & ~(_BV(7) | _BV(6))) | newBits;
  SREG = oldSREG;
}

// ── Motor primitive (independent PWMs) ───────────────────
// Writes two signed PWMs, one per motor. Zero on a motor
// leaves it in coast mode (enable on, brake released, PWM 0)
// so the belt can move that pulley freely.
// If both PWMs are 0, full brake.
void driveMotorSplit(int pwmA, int pwmB) {
  if (pwmA == 0 && pwmB == 0) {
    analogWrite(MOTOR_A_SV, 0);
    analogWrite(MOTOR_B_SV, 0);
    digitalWrite(MOTOR_A_BK, LOW);
    digitalWrite(MOTOR_B_BK, LOW);
    digitalWrite(MOTOR_A_EN, HIGH);
    digitalWrite(MOTOR_B_EN, HIGH);
    return;
  }

  digitalWrite(MOTOR_A_EN, LOW);
  digitalWrite(MOTOR_B_EN, LOW);
  digitalWrite(MOTOR_A_BK, HIGH);
  digitalWrite(MOTOR_B_BK, HIGH);

  bool aForward = (pwmA >= 0);
  bool bForward = (pwmB >= 0);
  setBothDirections(aForward, bForward);

  analogWrite(MOTOR_A_SV, (uint8_t)constrain(abs(pwmA), 0, 255));
  analogWrite(MOTOR_B_SV, (uint8_t)constrain(abs(pwmB), 0, 255));
}

// ── Motor primitive (wrapper) ────────────────────────────
// For small commands below DIAGONAL_THRESHOLD, drops motor B
// to coast-zero and runs motor A at MIN_SPEED — diagonal mode.
// For normal commands, both motors get the same PWM.
void driveMotor(int pwm) {
  if (pwm == 0) {
    driveMotorSplit(0, 0);
    return;
  }

  if (abs(pwm) < DIAGONAL_THRESHOLD) {
    int dir = (pwm > 0) ? 1 : -1;
    driveMotorSplit(dir * (int)MIN_SPEED, 0);
  } else {
    driveMotorSplit(pwm, pwm);
  }
}

// ── Brake ────────────────────────────────────────────────
void brakeAll() {
  lastCommandDir = 0;
  kicking        = false;
  ramping        = false;
  analogWrite(MOTOR_A_SV, 0);
  digitalWrite(MOTOR_A_BK, LOW);
  digitalWrite(MOTOR_A_EN, HIGH);
  analogWrite(MOTOR_B_SV, 0);
  digitalWrite(MOTOR_B_BK, LOW);
  digitalWrite(MOTOR_B_EN, HIGH);
}

// ── Pulse rate measurement ───────────────────────────────
static float measurePPS() {
  noInterrupts();
  unsigned long count = pulseCount;
  interrupts();
  unsigned long now = micros();

  unsigned long elapsed = now - kickWindowStart;
  if (elapsed < (unsigned long)KICK_MEASURE_US) return 0.0f;

  float pps = (float)((unsigned long)(count - kickWindowCount))
              * 1000000.0f / (float)elapsed;

  kickWindowStart  = now;
  kickWindowCount  = count;

  return pps;
}

// ── Kick start helper ────────────────────────────────────
static void startKick() {
  kicking    = true;
  ramping    = false;
  kickStartMs = millis();

  noInterrupts();
  kickWindowCount = pulseCount;
  interrupts();
  kickWindowStart = micros();
}

// ── Setup ────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(MOTOR_A_SV,  OUTPUT);
  pinMode(MOTOR_A_EN,  OUTPUT);
  pinMode(MOTOR_A_FR,  OUTPUT);
  pinMode(MOTOR_A_BK,  OUTPUT);
  pinMode(MOTOR_A_ALM, INPUT_PULLUP);
  pinMode(MOTOR_A_PG,  INPUT_PULLUP);

  pinMode(MOTOR_B_SV,  OUTPUT);
  pinMode(MOTOR_B_EN,  OUTPUT);
  pinMode(MOTOR_B_FR,  OUTPUT);
  pinMode(MOTOR_B_BK,  OUTPUT);
  pinMode(MOTOR_B_ALM, INPUT_PULLUP);
  pinMode(MOTOR_B_PG,  INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(MOTOR_A_PG), onPulseA, FALLING);

  brakeAll();
  lastMsgTime = millis();
}

// ── Main loop ────────────────────────────────────────────
void loop() {
  pollSerial();

  // ── Watchdog ────────────────────────────────────────────
  if (millis() - lastMsgTime > (unsigned long)WATCHDOG_MS) {
    brakeAll();
    return;
  }

  // ── Deadband ────────────────────────────────────────────
  int pwm = commandedPWM;
  if (abs(pwm) < PWM_DEADBAND) {
    if (lastCommandDir != 0) brakeAll();
    lastCommandDir = 0;
    return;
  }

  // Clamp to limits (no MIN_SPEED floor here — driveMotor()
  // handles the sub-threshold case via diagonal mode)
  pwm = constrain(pwm, -MAX_SPEED, MAX_SPEED);

  int newDir = (pwm > 0) ? 1 : -1;

  // ── Kick trigger ─────────────────────────────────────────
  // Fire only when starting from rest (lastCommandDir == 0).
  // Direction reversals go straight to commanded PWM.
  if (lastCommandDir == 0) {
    startKick();
  }
  lastCommandDir = newDir;

  // ── Kick phase ───────────────────────────────────────────
  if (kicking) {
    unsigned long now = millis();
    bool timeout      = (now - kickStartMs >= (unsigned long)KICK_MAX_MS);
    bool atSpeed      = (KICK_TARGET_PPS > 0 && measurePPS() >= (float)KICK_TARGET_PPS);

    if (timeout || atSpeed) {
      // Kick done — start ramp toward commanded PWM
      kicking      = false;
      ramping      = (KICK_RAMP_MS > 0);
      rampStartMs  = now;
      rampStartPWM = newDir * KICK_PWM;
    } else {
      // Still kicking — kick uses both motors at KICK_PWM
      driveMotorSplit(newDir * KICK_PWM, newDir * KICK_PWM);
      return;
    }
  }

  // ── Ramp phase ───────────────────────────────────────────
  // Linearly interpolate from rampStartPWM to commanded PWM.
  if (ramping) {
    unsigned long elapsed = millis() - rampStartMs;

    if (elapsed >= (unsigned long)KICK_RAMP_MS) {
      ramping = false;
    } else {
      float t      = (float)elapsed / (float)KICK_RAMP_MS;
      int   ramped = (int)(rampStartPWM + t * (float)(pwm - rampStartPWM));
      driveMotor(ramped);
      return;
    }
  }

  // ── Normal drive ─────────────────────────────────────────
  driveMotor(pwm);
}