#define THUMB  34
#define INDEX  35
#define MIDDLE 32
#define RING   33
#define LITTLE 25

#define THRESHOLD 2000  // Adjust if needed for ESP32 ADC scale

void setup() {
  Serial.begin(115200);  // ESP32 usually works better with 115200
  Serial.println("Sign Language Recognition Started on ESP32");
}

void loop() {

  int t = analogRead(THUMB);
  int i = analogRead(INDEX);
  int m = analogRead(MIDDLE);
  int r = analogRead(RING);
  int l = analogRead(LITTLE);

  bool T = t > THRESHOLD;
  bool I = i > THRESHOLD;
  bool M = m > THRESHOLD;
  bool R = r > THRESHOLD;
  bool L = l > THRESHOLD;

  char output = '-';

  // ===== LETTERS A–Z =====
  if (!T && I && M && R && L) output = 'A';
  else if (T && !I && !M && !R && !L) output = 'B';
  else if (T && I && !M && !R && L) output = 'C';
  else if (!T && !I && M && R && L) output = 'D';
  else if (T && !I && M && R && L) output = 'E';
  else if (!T && !I && !M && R && L) output = 'F';
  else if (!T && !I && !M && !R && L) output = 'G';
  else if (T && I && M && !R && !L) output = 'I';
  else if (!T && I && !M && R && !L) output = 'J';
  else if (!T && !I && M && !R && L) output = 'K';
  else if (T && !I && !M && R && !L) output = 'L';
  else if (!T && I && M && !R && L) output = 'M';
  else if (T && !I && M && !R && L) output = 'N';
  else if (!T && I && M && R && !L) output = 'O';
  else if (T && !I && !M && !R && L) output = 'P';
  else if (!T && !I && M && R && !L) output = 'Q';
  else if (T && I && !M && R && !L) output = 'R';
  else if (!T && I && !M && !R && L) output = 'S';
  else if (T && !I && M && !R && !L) output = 'T';
  else if (!T && !I && !M && R && !L) output = 'U';
  else if (T && I && M && R && !L) output = 'V';
  else if (!T && I && M && R && L) output = 'W';
  else if (T && !I && M && R && !L) output = 'X';
  else if (!T && !I && M && !R && !L) output = 'Y';
  else if (T && I && !M && !R && !L) output = 'Z';

  // ===== NUMBERS 0–9 =====
  else if (!T && !I && !M && !R && !L) output = '0';
  else if (T && !I && !M && !R && !L) output = '1';
  else if (!T && I && !M && !R && !L) output = '2';
  else if (!T && !I && M && !R && !L) output = '3';
  else if (!T && !I && !M && R && !L) output = '4';
  else if (!T && !I && !M && !R && L) output = '5';
  else if (T && I && !M && !R && !L) output = '6';
  else if (T && !I && M && !R && !L) output = '7';
  else if (T && !I && !M && R && !L) output = '8';
  else if (T && !I && !M && !R && L) output = '9';

  // ===== SERIAL OUTPUT =====
  Serial.print("Thumb: "); Serial.print(t);
  Serial.print(" | Index: "); Serial.print(i);
  Serial.print(" | Middle: "); Serial.print(m);
  Serial.print(" | Ring: "); Serial.print(r);
  Serial.print(" | Little: "); Serial.print(l);
  Serial.print("  ==> Output: ");
  Serial.println(output);

  delay(500);
} 