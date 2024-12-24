#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <SD.h>
#include <esp_adc_cal.h>
#include <SSD1306Wire.h>
#include "OLEDDisplayUi.h"
#include <RadioLib.h>
#include "utilities.h"
#include <AceButton.h>

#ifdef HAS_DISPLAY
#include <U8g2lib.h>

#ifndef DISPLAY_MODEL
#define DISPLAY_MODEL U8G2_SSD1306_128X64_NONAME_F_HW_I2C
#endif

DISPLAY_MODEL *u8g2 = nullptr;
#endif

using namespace ace_button;

bool readkey();
void radioTx(OLEDDisplay *display, OLEDDisplayUiState *state, int16_t x, int16_t y);
void radioRx(OLEDDisplay *display, OLEDDisplayUiState *state, int16_t x, int16_t y);
void hwInfo(OLEDDisplay *display, OLEDDisplayUiState *state, int16_t x, int16_t y);
void setFlag(void);

#if defined(USING_SX1268_433M)
uint8_t txPower = 22;
float radioFreq = 433.0;
SX1268 radio = new Module(RADIO_CS_PIN, RADIO_DIO1_PIN, RADIO_RST_PIN, RADIO_BUSY_PIN);

#elif defined(USING_SX1262_868M)
uint8_t txPower = 22;
float radioFreq = 868.0;
SX1262 radio = new Module(RADIO_CS_PIN, RADIO_DIO1_PIN, RADIO_RST_PIN, RADIO_BUSY_PIN);

#endif

// save transmission state between loops
int transmissionState = RADIOLIB_ERR_NONE;
bool transmittedFlag = false;
volatile bool enableInterrupt = true;
uint32_t transmissionCounter = 0;
uint32_t recvCounter = 0;
float radioRSSI = 0;

bool isRadioOnline = false;
bool isWifiOnline = false;
bool isSdCardOnline = false;
bool rxStatus = true;
uint32_t radioRunInterval = 0;
uint32_t batteryRunInterval = 0;

SSD1306Wire display(0x3c, I2C_SDA, I2C_SCL);
OLEDDisplayUi ui(&display);
FrameCallback frames[] = {hwInfo, radioTx, radioRx};
SPIClass SDSPI(HSPI);
AceButton button;

void setFlag(void)
{
    // check if the interrupt is enabled
    if (!enableInterrupt)
    {
        return;
    }
    // we got a packet, set the flag
    transmittedFlag = true;
}

void handleEvent(AceButton *button, uint8_t eventType, uint8_t buttonState)
{
    int state;
    static uint8_t framCounter = 1;
    switch (eventType)
    {
    case AceButton::kEventClicked:
        Serial.printf("framCounter : %d\n", framCounter);
        switch (framCounter)
        {
        case 0:
            enableInterrupt = false;
            break;
        case 1:
            enableInterrupt = true;
            Serial.println("Start transmit");
            state = radio.transmit((uint8_t *)&transmissionCounter, 4);
            if (state != RADIOLIB_ERR_NONE)
            {
                Serial.println(F("[Radio] transmit packet failed!"));
            }
            break;
        case 2:
            enableInterrupt = true;
            Serial.println("Start receive");
            state = radio.startReceive();
            if (state != RADIOLIB_ERR_NONE)
            {
                Serial.println(F("[Radio] Received packet failed!"));
            }
            break;
        default:
            break;
        }
        framCounter++;
        ui.nextFrame();
        framCounter %= 3;
        break;
    case AceButton::kEventLongPressed:
        Serial.println("sleep test");
        Wire.beginTransmission(0x3C);
        if (Wire.endTransmission() == 0)
        {
            u8g2 = new DISPLAY_MODEL(U8G2_R0, U8X8_PIN_NONE);
            u8g2->begin();
            u8g2->clearBuffer();
            u8g2->setFlipMode(0);
            u8g2->setFontMode(1);
            u8g2->setDrawColor(1);
            u8g2->setFontDirection(0);
            u8g2->firstPage();
            u8g2->setFont(u8g2_font_fur11_tf);
        }

        WiFi.enableAP(false);
        u8g2->sleepOn();

        radio.sleep(false);
        SPI.end();

        // SD.end();
        SDSPI.end();

        pinMode(RADIO_RST_PIN, INPUT_PULLUP);
        pinMode(RADIO_BUSY_PIN, INPUT_PULLUP);
        pinMode(RADIO_DIO1_PIN, INPUT_PULLUP);

        pinMode(RADIO_CS_PIN, OUTPUT);
        digitalWrite(RADIO_CS_PIN, HIGH);
        pinMode(RADIO_SCLK_PIN, OUTPUT);
        digitalWrite(RADIO_SCLK_PIN, HIGH);
        pinMode(RADIO_MOSI_PIN, OUTPUT);
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

        esp_sleep_enable_timer_wakeup(30 * 1000 * 1000);
        esp_deep_sleep_start();
        break;
    }
}

void gpio_test()
{
    int gpioStatus = LOW;

    /* GPIO_PIN_38(INPUT) <-> GPIO_PIN_16(OUTPUT) */
    /* GPIO_PIN_39(INPUT) <-> GPIO_PIN_15(OUTPUT) */
    /* GPIO_PIN_40(INPUT) <-> GPIO_PIN_12(OUTPUT) */
    /* GPIO_PIN_41(INPUT) <-> GPIO_PIN_48(OUTPUT) */
    /* GPIO_PIN_45(INPUT) <-> GPIO_PIN_47(OUTPUT) */
    /* GPIO_PIN_46(INPUT) <-> GPIO_PIN_35(OUTPUT) */
    /* GPIO_PIN_42(INPUT) <-> GPIO_PIN_36(OUTPUT) */

    pinMode(GPIO_PIN_15, OUTPUT);
    pinMode(GPIO_PIN_16, OUTPUT);
    pinMode(I2C_SCL, OUTPUT);
    pinMode(SDCARD_MOSI, OUTPUT);
    pinMode(GPIO_PIN_12, OUTPUT);
    pinMode(SDCARD_CS, OUTPUT);
    pinMode(RADIO_DIO1_PIN, OUTPUT);
    pinMode(GPIO_PIN_47, OUTPUT);
    pinMode(GPIO_PIN_48, OUTPUT);
    pinMode(RADIO_BUSY_PIN, OUTPUT);
    pinMode(GPIO_PIN_35, OUTPUT);
    pinMode(GPIO_PIN_36, OUTPUT);
    pinMode(GPIO_PIN_37, OUTPUT);

    digitalWrite(GPIO_PIN_15, LOW);
    digitalWrite(GPIO_PIN_16, HIGH);
    digitalWrite(I2C_SCL, LOW);
    digitalWrite(SDCARD_MOSI, LOW);
    digitalWrite(GPIO_PIN_12, HIGH);
    digitalWrite(SDCARD_CS, LOW);
    digitalWrite(GPIO_PIN_48, LOW);
    digitalWrite(GPIO_PIN_47, HIGH);
    digitalWrite(RADIO_DIO1_PIN, LOW);
    digitalWrite(RADIO_BUSY_PIN, HIGH);
    digitalWrite(GPIO_PIN_35, LOW);
    digitalWrite(GPIO_PIN_36, HIGH);
    digitalWrite(GPIO_PIN_37, LOW);

    pinMode(GPIO_PIN_38, INPUT_PULLUP);
    pinMode(GPIO_PIN_39, INPUT_PULLUP);
    pinMode(GPIO_PIN_40, INPUT_PULLUP);
    pinMode(GPIO_PIN_41, INPUT_PULLUP);
    pinMode(GPIO_PIN_45, INPUT_PULLUP);
    pinMode(GPIO_PIN_46, INPUT_PULLUP);
    pinMode(GPIO_PIN_42, INPUT_PULLUP);

    /* GPIO_PIN_38(INPUT) <-> GPIO_PIN_16(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_38);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO38<->GPIO16 error !");
    }
    /* GPIO_PIN_39(INPUT) <-> GPIO_PIN_15(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_39);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO39<->GPIO15 error !");
    }
    /* GPIO_PIN_40(INPUT) <-> GPIO_PIN_12(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_40);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO40<->GPIO12 error !");
    }
    /* GPIO_PIN_41(INPUT) <-> GPIO_PIN_48(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_41);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO41<->GPIO48 error !");
    }
    /* GPIO_PIN_45(INPUT) <-> GPIO_PIN_47(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_45);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO45<->GPIO47 error !");
    }
    /* GPIO_PIN_46(INPUT) <-> GPIO_PIN_35(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_46);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO46<->GPIO35 error !");
    }
    /* GPIO_PIN_42(INPUT) <-> GPIO_PIN_36(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_42);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO42<->GPIO36 error !");
    }

    digitalWrite(GPIO_PIN_15, HIGH);
    digitalWrite(GPIO_PIN_16, LOW);
    digitalWrite(I2C_SCL, HIGH);
    digitalWrite(SDCARD_MOSI, HIGH);
    digitalWrite(GPIO_PIN_12, LOW);
    digitalWrite(SDCARD_CS, HIGH);
    digitalWrite(GPIO_PIN_48, HIGH);
    digitalWrite(GPIO_PIN_47, LOW);
    digitalWrite(RADIO_DIO1_PIN, HIGH);
    digitalWrite(RADIO_BUSY_PIN, LOW);
    digitalWrite(GPIO_PIN_35, HIGH);
    digitalWrite(GPIO_PIN_36, LOW);
    digitalWrite(GPIO_PIN_37, HIGH);

    /* GPIO_PIN_38(INPUT) <-> GPIO_PIN_16(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_38);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO38<->GPIO16 error !");
    }
    /* GPIO_PIN_39(INPUT) <-> GPIO_PIN_15(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_39);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO39<->GPIO15 error !");
    }
    /* GPIO_PIN_40(INPUT) <-> GPIO_PIN_12(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_40);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO40<->GPIO12 error !");
    }
    /* GPIO_PIN_41(INPUT) <-> GPIO_PIN_48(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_41);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO41<->GPIO48 error !");
    }
    /* GPIO_PIN_45(INPUT) <-> GPIO_PIN_47(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_45);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO45<->GPIO47 error !");
    }
    /* GPIO_PIN_46(INPUT) <-> GPIO_PIN_35(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_46);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO46<->GPIO35 error !");
    }
    /* GPIO_PIN_42(INPUT) <-> GPIO_PIN_36(OUTPUT) */
    gpioStatus = digitalRead(GPIO_PIN_42);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO42<->GPIO36 error !");
    }

    /* GPIO_PIN_38(OUTPUT) <-> GPIO_PIN_16(INPUT) */
    /* GPIO_PIN_39(OUTPUT) <-> GPIO_PIN_15(INPUT) */
    /* GPIO_PIN_40(OUTPUT) <-> GPIO_PIN_12(INPUT) */
    /* GPIO_PIN_41(OUTPUT) <-> GPIO_PIN_48(INPUT) */
    /* GPIO_PIN_45(OUTPUT) <-> GPIO_PIN_47(INPUT) */
    /* GPIO_PIN_46(OUTPUT) <-> GPIO_PIN_35(INPUT) */
    /* GPIO_PIN_42(OUTPUT) <-> GPIO_PIN_36(INPUT) */

    pinMode(GPIO_PIN_38, OUTPUT);
    pinMode(GPIO_PIN_39, OUTPUT);
    pinMode(GPIO_PIN_40, OUTPUT);
    pinMode(GPIO_PIN_41, OUTPUT);
    pinMode(GPIO_PIN_42, OUTPUT);
    pinMode(GPIO_PIN_45, OUTPUT);
    pinMode(GPIO_PIN_46, OUTPUT);

    digitalWrite(GPIO_PIN_38, LOW);
    digitalWrite(GPIO_PIN_39, HIGH);
    digitalWrite(GPIO_PIN_40, LOW);
    digitalWrite(GPIO_PIN_41, LOW);
    digitalWrite(GPIO_PIN_42, HIGH);
    digitalWrite(GPIO_PIN_45, LOW);
    digitalWrite(GPIO_PIN_46, HIGH);

    pinMode(GPIO_PIN_16, INPUT_PULLUP);
    pinMode(GPIO_PIN_15, INPUT_PULLUP);
    pinMode(GPIO_PIN_12, INPUT_PULLUP);
    pinMode(GPIO_PIN_48, INPUT_PULLUP);
    pinMode(GPIO_PIN_47, INPUT_PULLUP);
    pinMode(GPIO_PIN_35, INPUT_PULLUP);
    pinMode(GPIO_PIN_36, INPUT_PULLUP);

    /* GPIO_PIN_38(OUTPUT) <-> GPIO_PIN_16(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_16);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO38<->GPIO16 error !");
    }
    /* GPIO_PIN_39(OUTPUT) <-> GPIO_PIN_15(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_15);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO39<->GPIO15 error !");
    }
    /* GPIO_PIN_40(OUTPUT) <-> GPIO_PIN_12(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_12);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO40<->GPIO12 error !");
    }
    /* GPIO_PIN_41(OUTPUT) <-> GPIO_PIN_48(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_48);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO41<->GPIO48 error !");
    }
    /* GPIO_PIN_45(OUTPUT) <-> GPIO_PIN_47(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_47);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO45<->GPIO47 error !");
    }
    /* GPIO_PIN_46(OUTPUT) <-> GPIO_PIN_35(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_35);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO46<->GPIO35 error !");
    }
    /* GPIO_PIN_42(OUTPUT) <-> GPIO_PIN_36(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_36);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO42<->GPIO36 error !");
    }

    digitalWrite(GPIO_PIN_38, HIGH);
    digitalWrite(GPIO_PIN_39, LOW);
    digitalWrite(GPIO_PIN_40, HIGH);
    digitalWrite(GPIO_PIN_41, HIGH);
    digitalWrite(GPIO_PIN_42, LOW);
    digitalWrite(GPIO_PIN_45, HIGH);
    digitalWrite(GPIO_PIN_46, LOW);

    /* GPIO_PIN_38(OUTPUT) <-> GPIO_PIN_16(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_16);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO38<->GPIO16 error !");
    }
    /* GPIO_PIN_39(OUTPUT) <-> GPIO_PIN_15(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_15);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO39<->GPIO15 error !");
    }
    /* GPIO_PIN_40(OUTPUT) <-> GPIO_PIN_12(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_12);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO40<->GPIO12 error !");
    }
    /* GPIO_PIN_41(OUTPUT) <-> GPIO_PIN_48(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_48);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO41<->GPIO48 error !");
    }
    /* GPIO_PIN_45(OUTPUT) <-> GPIO_PIN_47(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_47);
    if (gpioStatus != HIGH)
    {
        Serial.println("GPIO45<->GPIO47 error !");
    }
    /* GPIO_PIN_46(OUTPUT) <-> GPIO_PIN_35(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_35);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO46<->GPIO35 error !");
    }
    /* GPIO_PIN_42(OUTPUT) <-> GPIO_PIN_36(INPUT) */
    gpioStatus = digitalRead(GPIO_PIN_36);
    if (gpioStatus != LOW)
    {
        Serial.println("GPIO42<->GPIO36 error !");
    }

    pinMode(I2C_SCL, INPUT_PULLUP);
    pinMode(SDCARD_MOSI, INPUT_PULLUP);
    pinMode(SDCARD_CS, INPUT_PULLUP);
    pinMode(RADIO_DIO1_PIN, INPUT_PULLUP);
    pinMode(RADIO_BUSY_PIN, INPUT_PULLUP);
    pinMode(GPIO_PIN_37, INPUT_PULLUP);
    pinMode(GPIO_PIN_38, INPUT_PULLUP);
    pinMode(GPIO_PIN_39, INPUT_PULLUP);
    pinMode(GPIO_PIN_40, INPUT_PULLUP);
    pinMode(GPIO_PIN_41, INPUT_PULLUP);
    pinMode(GPIO_PIN_42, INPUT_PULLUP);
    pinMode(GPIO_PIN_45, INPUT_PULLUP);
    pinMode(GPIO_PIN_46, INPUT_PULLUP);
}

void setup()
{
    Serial.begin(115200);
    Serial.println("initBoard");

    gpio_test();
    pinMode(BOARD_LED, OUTPUT);
    digitalWrite(BOARD_LED, LED_ON);

    Wire.begin(I2C_SDA, I2C_SCL);

    SDSPI.begin(SDCARD_SCLK, SDCARD_MISO, SDCARD_MOSI, SDCARD_CS);
    isSdCardOnline = SD.begin(SDCARD_CS, SDSPI);
    if (!isSdCardOnline)
    {
        Serial.println("setupSDCard FAIL");
    }
    else
    {
        uint32_t cardSize = SD.cardSize() / (1024 * 1024);
        Serial.print("setupSDCard PASS . SIZE = ");
        Serial.print(cardSize);
        Serial.println(" MB");
    }

    pinMode(BUTTON_PIN, INPUT_PULLUP);
    button.init(BUTTON_PIN);
    ButtonConfig *buttonConfig = button.getButtonConfig();
    buttonConfig->setEventHandler(handleEvent);
    buttonConfig->setFeature(ButtonConfig::kFeatureClick);
    buttonConfig->setFeature(ButtonConfig::kFeatureLongPress);

    // Initialising the UI will init the display too.
    ui.setTargetFPS(60);

    // You can change this to
    // TOP, LEFT, BOTTOM, RIGHT
    ui.setIndicatorPosition(BOTTOM);

    // Defines where the first frame is located in the bar.
    ui.setIndicatorDirection(LEFT_RIGHT);

    // You can change the transition that is used
    // SLIDE_LEFT, SLIDE_RIGHT, SLIDE_UP, SLIDE_DOWN
    ui.setFrameAnimation(SLIDE_LEFT);

    // Add frames
    ui.setFrames(frames, sizeof(frames) / sizeof(frames[0]));

    ui.disableAutoTransition();

    // Initialising the UI will init the display too.
    ui.init();

    display.flipScreenVertically();

    Serial.print(F("[WiFi] Initializing ... "));
    int wifi_state = WiFi.mode(WIFI_AP);
    isWifiOnline = wifi_state == true;
    if (wifi_state == true)
    {
        Serial.println(F("success!"));
    }
    else
    {
        Serial.println(F("failed!"));
    }
    WiFi.softAP("ESP32-S3-AP", "12345678");

    Serial.println("WiFi SSID : ESP32-S3-AP");
    Serial.println("WiFi PASS : 12345678 ");

    SPI.begin(RADIO_SCLK_PIN, RADIO_MISO_PIN, RADIO_MOSI_PIN);
    Serial.print(F("[Radio] Initializing ... "));
    int state = radio.begin(radioFreq);

    if (state == RADIOLIB_ERR_NONE)
    {
        Serial.println(F("success!"));
    }
    else
    {
        Serial.println(F("failed!"));
    }
    isRadioOnline = state == RADIOLIB_ERR_NONE;
    enableInterrupt = false;

    // modules are standard transmission power.
    if (radio.setOutputPower(txPower) == RADIOLIB_ERR_INVALID_OUTPUT_POWER)
    {
        Serial.println(F("Selected output power is invalid for this module!"));
    }

    // set bandwidth to 250 kHz
    if (radio.setBandwidth(125.0) == RADIOLIB_ERR_INVALID_BANDWIDTH)
    {
        Serial.println(F("Selected bandwidth is invalid for this module!"));
    }

    // set over current protection limit to 80 mA (accepted range is 45 - 240 mA)
    // NOTE: set value to 0 to disable overcurrent protection
    if (radio.setCurrentLimit(140) == RADIOLIB_ERR_INVALID_CURRENT_LIMIT)
    {
        Serial.println(F("Selected current limit is invalid for this module!"));
    }

    // set the function that will be called
    // when new packet is received
    radio.setDio1Action(setFlag);
    // radio.setDio2AsRfSwitch(true);

    // start listening for LoRa packets
    Serial.print(F("[Radio] Starting to listen ... "));
    state = radio.startReceive();
    if (state != RADIOLIB_ERR_NONE)
    {
        Serial.println(F("[Radio] Received packet failed!"));
    }
    // 设置spreadfactor，这需要根据你的具体需求和模块规格来确定
    radio.setSpreadingFactor(7.0);
}

void loop()
{
    button.check();
    ui.update();
    delay(1);
}

void radioTx(OLEDDisplay *display, OLEDDisplayUiState *state, int16_t x, int16_t y)
{
    display->setTextAlignment(TEXT_ALIGN_LEFT);
    if (millis() - radioRunInterval > 1000)
    {

        if (transmittedFlag)
        {
            // reset flag
            transmittedFlag = false;
            if (transmissionState == RADIOLIB_ERR_NONE)
            {
                // packet was successfully sent
                Serial.println(F("transmission finished!"));

                // NOTE: when using interrupt-driven transmit method,
                //       it is not possible to automatically measure
                //       transmission data rate using getDataRate()
            }
            else
            {
                Serial.print(F("failed, code "));
                Serial.println(transmissionState);
            }

            // clean up after transmission is finished
            // this will ensure transmitter is disabled,
            // RF switch is powered down etc.
            radio.finishTransmit();

            // send another one
            Serial.print(F("[Radio] Sending another packet ... "));

            // you can transmit C-string or Arduino string up to
            // 256 characters long
            // transmissionState = radio.startTransmit("Hello World!");
            radio.transmit((uint8_t *)&transmissionCounter, 4);
            transmissionCounter++;
            radioRunInterval = millis();

            // you can also transmit byte array up to 256 bytes long
            /*
              byte byteArr[] = {0x01, 0x23, 0x45, 0x67,
                                0x89, 0xAB, 0xCD, 0xEF};
              int state = radio.startTransmit(byteArr, 8);
            */
            digitalWrite(BOARD_LED, 1 - digitalRead(BOARD_LED));
        }
        Serial.println("Radio TX done !");
    }

    display->drawString(0 + x, 0 + y, "RADIO TX ...");
    display->drawString(0 + x, 16 + y, "TX :" + String(transmissionCounter));
}

void radioRx(OLEDDisplay *display, OLEDDisplayUiState *state, int16_t x, int16_t y)
{
    display->setFont(ArialMT_Plain_10);
    // The coordinates define the left starting point of the text
    display->setTextAlignment(TEXT_ALIGN_LEFT);

    // check if the flag is set
    if (transmittedFlag)
    {
        Serial.println("Radio RX done !");

        digitalWrite(BOARD_LED, 1 - digitalRead(BOARD_LED));

        // disable the interrupt service routine while
        // processing the data
        enableInterrupt = false;

        // reset flag
        transmittedFlag = false;

        // you can read received data as an Arduino String
        int state = radio.readData((uint8_t *)&recvCounter, 4);

        // you can also read received data as byte array
        /*
          byte byteArr[8];
          int state = radio.readData(byteArr, 8);
        */

        if (state == RADIOLIB_ERR_NONE)
        {
            // packet was successfully received
            Serial.println(F("[Radio] Received packet!"));
            radioRSSI = radio.getRSSI();
        }
        else if (state == RADIOLIB_ERR_CRC_MISMATCH)
        {
            // packet was received, but is malformed
            Serial.println(F("[Radio] CRC error!"));
        }
        else
        {
            // some other error occurred
            Serial.print(F("[Radio] Failed, code "));
            Serial.println(state);
        }

        // put module back to listen mode
        radio.startReceive();

        // we're ready to receive more packets,
        // enable interrupt service routine
        enableInterrupt = true;
    }

    display->drawString(0 + x, 0 + y, "RADIO RX ...");
    display->drawString(0 + x, 12 + y, "RX :" + String(recvCounter));
    display->drawString(0 + x, 24 + y, "RSSI:" + String(radioRSSI));
}

void hwInfo(OLEDDisplay *display, OLEDDisplayUiState *state, int16_t x, int16_t y)
{
    static char buffer[64];
    if (millis() - batteryRunInterval > 1000)
    {
        esp_adc_cal_characteristics_t adc_chars;
        esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, &adc_chars);
        uint16_t raw = analogRead(BAT_ADC_PIN);
        float volotage = (float)(esp_adc_cal_raw_to_voltage(raw, &adc_chars) * 2) / 1000.0;
        sprintf(buffer, "%.2fV", volotage);
        batteryRunInterval = millis();
    }

    display->setFont(ArialMT_Plain_10);
    display->setTextAlignment(TEXT_ALIGN_LEFT);
    display->drawString(0 + x, 0 + y, "RADIO  ");
    display->drawString(50 + x, 0 + y, isRadioOnline & 1 ? "+" : "NA");
    display->drawString(0 + x, 12 + y, "SD   ");
    display->drawString(50 + x, 12 + y, isSdCardOnline & 1 ? "+" : "NA");
    display->drawString(0 + x, 24 + y, "BAT   ");
    display->drawString(50 + x, 24 + y, buffer);
}
