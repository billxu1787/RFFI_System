#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include "utilities.h"

#ifdef HAS_SDCARD
#include <SD.h>
#include <FS.h>
#endif

#ifdef HAS_DISPLAY
#include <U8g2lib.h>

#ifndef DISPLAY_MODEL
#define DISPLAY_MODEL U8G2_SSD1306_128X64_NONAME_F_HW_I2C
#endif

DISPLAY_MODEL *u8g2 = nullptr;
#endif

#ifndef OLED_WIRE_PORT
#define OLED_WIRE_PORT Wire
#endif

#define initPMU()
#define disablePeripherals()

SPIClass SDSPI(HSPI);

void initBoard()
{
    Serial.begin(115200);
    Serial.println("initBoard");
    SPI.begin(RADIO_SCLK_PIN, RADIO_MISO_PIN, RADIO_MOSI_PIN);
    Wire.begin(I2C_SDA, I2C_SCL);

#if OLED_RST
    pinMode(OLED_RST, OUTPUT);
    digitalWrite(OLED_RST, HIGH);
    delay(20);
    digitalWrite(OLED_RST, LOW);
    delay(20);
    digitalWrite(OLED_RST, HIGH);
    delay(20);
#endif

    initPMU();

#ifdef BOARD_LED
    pinMode(BOARD_LED, OUTPUT);
    digitalWrite(BOARD_LED, LED_ON);
#endif
}
