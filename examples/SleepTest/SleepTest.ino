

#include <RadioLib.h>
#include "boards.h"

#if defined(USING_SX1268_433M)
uint8_t txPower = 22;
float radioFreq = 433.0;
SX1268 radio = new Module(RADIO_CS_PIN, RADIO_DIO1_PIN, RADIO_RST_PIN, RADIO_BUSY_PIN);

#elif defined(USING_SX1262_868M)
uint8_t txPower = 22;
float radioFreq = 868.0;
SX1262 radio = new Module(RADIO_CS_PIN, RADIO_DIO1_PIN, RADIO_RST_PIN, RADIO_BUSY_PIN);

#endif

void setup()
{
    initBoard();

    // initialize SX126x with default settings
    Serial.print(F("[SX126x] Initializing ... "));
    int state = radio.begin(radioFreq);

#ifdef HAS_DISPLAY
    if (u8g2) {
        if (state != RADIOLIB_ERR_NONE) {
            u8g2->drawStr(0, 32, "Initializing: FAIL!");
            u8g2->sendBuffer();
        }
        else{
            u8g2->drawStr(0, 32, "Initializing: OK!");
            u8g2->sendBuffer();
        }
        delay(1000);
    }
#endif
    if (state == RADIOLIB_ERR_NONE) {
        Serial.println(F("success!"));
        Serial.print(F("[Radio] Starting to listen ... "));
         state = radio.startReceive();
        if (state != RADIOLIB_ERR_NONE) {
            Serial.println(F("[Radio] Received packet failed!"));
        }
    } else {
        Serial.print(F("failed, code "));
        Serial.println(state);
         while (true);  
    }

    Serial.println("sleep test");

    u8g2->sleepOn();
    radio.sleep(false);    

    SPI.end();
    SDSPI.end();

    pinMode(RADIO_RST_PIN, INPUT_PULLUP);
    pinMode(RADIO_BUSY_PIN, INPUT_PULLUP);
    pinMode(RADIO_DIO1_PIN, INPUT_PULLUP);

    pinMode(RADIO_CS_PIN,OUTPUT);
    digitalWrite(RADIO_CS_PIN, HIGH);
    pinMode(RADIO_SCLK_PIN,OUTPUT);
    digitalWrite(RADIO_SCLK_PIN, HIGH);
    pinMode(RADIO_MOSI_PIN,OUTPUT);
    digitalWrite(RADIO_MOSI_PIN, HIGH);
    pinMode(RADIO_MISO_PIN, INPUT_PULLUP);

    pinMode(I2C_SDA, INPUT_PULLUP);
    pinMode(I2C_SCL, INPUT_PULLUP);

    pinMode(SDCARD_MOSI, INPUT_PULLUP);
    pinMode(SDCARD_MISO, INPUT_PULLUP);
    pinMode(SDCARD_SCLK, INPUT_PULLUP);
    pinMode(SDCARD_CS, INPUT_PULLUP);

    pinMode(BOARD_LED, OUTPUT);
    digitalWrite(BOARD_LED, LOW);
    pinMode(BAT_ADC_PIN, INPUT_PULLUP);
    
    delay(2000);

    esp_sleep_enable_timer_wakeup(30 * 1000 * 1000);
    esp_deep_sleep_start();
}

void loop()
{
}
