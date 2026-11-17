/*
 * Two-ESP32 ESP-NOW CSI link with key-driven frequency hopping.
 * ROLE selects transmitter or receiver; HOP_ENABLE selects fixed channel or hopping.
 * Channel for step n = seq / HOP_DWELL_PKTS is chan_for_step(n) = 1 + (H(n, KEY) mod 13).
 */
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "rom/ets_sys.h"

/* ===================== CONFIG - EDIT THESE ========================= */
#define ROLE_TX   1
#define ROLE_RX   2
#define ROLE      ROLE_RX          /* <-- set ROLE_TX on one board, ROLE_RX on the other */

#define FIXED_CH        6          /* used only when HOP_ENABLE 0 */
#define SEND_HZ         100        /* packets per second from the transmitter */
#define HOP_ENABLE      1          /* 1 = key-driven 13-ch hopping (Phase B); 0 = fixed channel */


#if HOP_ENABLE
/* ===== KEY-DRIVEN HOPPING over all 13 channels ===================== */
#define SECRET_KEY        0xA5F301C7u          /* <-- shared secret, identical on both boards */
#define NUM_HOP_CH        13                   /* hop over channels 1..13 */
#define HOP_DWELL_MS      2000                  /* dwell time per channel */
#define HOP_DWELL_PKTS    (SEND_HZ * HOP_DWELL_MS / 1000)
#define FOLLOW_TIMEOUT_MS 120                   /* silence after which the RX moves to the next channel */

/* Keyed pseudo-random channel for hop-step n (a mixing hash of key and step) */
static inline uint8_t chan_for_step(uint32_t n)
{
    uint32_t x = (n * 2654435761u) ^ SECRET_KEY;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (uint8_t)(1 + (x % NUM_HOP_CH));     /* channel 1..13 */
}
#define START_CH          chan_for_step(0)
#else
#define START_CH          FIXED_CH
#endif

static const char *TAG = "csi_hop_pair";

/* broadcast address: the TX does not need to know the RX's MAC */
static const uint8_t BCAST[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

/* 32-byte payload. Only seq is used by the receiver (step = seq / HOP_DWELL_PKTS) */
typedef struct __attribute__((packed)) {
    uint32_t seq;          /* packet counter; defines the hop step */
    uint8_t  hop_index;    /* current step (low byte), for logging */
    uint8_t  next_index;   /* next step (low byte), for logging */
    uint8_t  pad[26];      /* keep every packet the same length */
} hop_pkt_t;

/* ---------------- shared Wi-Fi / ESP-NOW setup ---------------- */
static void wifi_common_init(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));   /* never batch frames */
    ESP_ERROR_CHECK(esp_wifi_set_channel(START_CH, WIFI_SECOND_CHAN_NONE));

    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "my MAC " MACSTR "  start channel %d", MAC2STR(mac), START_CH);
    ESP_LOGW(TAG, "write BOTH boards' MACs down - we need them for unicast later");
}

static void espnow_common_init(void)
{
    ESP_ERROR_CHECK(esp_now_init());
    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, BCAST, 6);
    peer.channel = 0;              /* 0 = current wifi channel, so hops take effect */
    peer.ifidx   = WIFI_IF_STA;
    peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    /* ESP-NOW defaults to 1 Mbps 802.11b, which gives no CSI; force 6 Mbps OFDM */
    esp_now_rate_config_t rate = {
        .phymode = WIFI_PHY_MODE_11G,
        .rate    = WIFI_PHY_RATE_6M,
        .ersu    = false,
        .dcm     = false,
    };
    esp_err_t re = esp_now_set_peer_rate_config(BCAST, &rate);
    if (re != ESP_OK) {
        ESP_LOGW(TAG, "set_peer_rate_config: %s (CSI may stay empty)",
                 esp_err_to_name(re));
    }
}

/* ========================== TRANSMITTER =========================== */
#if ROLE == ROLE_TX
static void tx_task(void *arg)
{
    hop_pkt_t p = {0};
    const TickType_t period = pdMS_TO_TICKS(1000 / SEND_HZ);
#if HOP_ENABLE
    uint8_t cur_ch = chan_for_step(0);   /* both boards start on step 0 */
    ESP_LOGI(TAG, "TX: %d pkt/s, KEY-HOPPING over channels 1..%d every %d ms",
             SEND_HZ, NUM_HOP_CH, HOP_DWELL_MS);
#else
    ESP_LOGI(TAG, "TX: sending %d pkt/s on ch %d (no hop)", SEND_HZ, FIXED_CH);
#endif
    while (1) {
#if HOP_ENABLE
        /* channel comes from the key and the step; the RX gets the step from seq */
        uint32_t step = p.seq / HOP_DWELL_PKTS;
        uint8_t  ch   = chan_for_step(step);
        if (ch != cur_ch) {
            cur_ch = ch;
            esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
            ets_printf("HOP,TX,%u,%u,%u\n", (unsigned)(step & 0xFF), (unsigned)ch,
                       (unsigned)(esp_timer_get_time() / 1000));
        }
        p.hop_index  = (uint8_t)(step & 0xFF);         /* for logging only */
        p.next_index = (uint8_t)((step + 1) & 0xFF);
#else
        p.hop_index  = 0;
        p.next_index = 0;
#endif
        esp_err_t e = esp_now_send(BCAST, (uint8_t *)&p, sizeof(p));
        if (e != ESP_OK && (p.seq % SEND_HZ) == 0) {
            ESP_LOGW(TAG, "esp_now_send: %s", esp_err_to_name(e));
        }
        p.seq++;
        vTaskDelay(period);
    }
}
#endif

/* =========================== RECEIVER ============================= */
#if ROLE == ROLE_RX
/* latest hop index heard from the TX */
static volatile uint8_t s_last_hop_index = 0;

/* counters printed once a second: packets received vs CSI produced */
static volatile uint32_t s_recv_count = 0;   /* ESP-NOW frames delivered */
static volatile uint32_t s_csi_count  = 0;   /* of those, how many gave CSI */

#if HOP_ENABLE
/* hop-follow state: step from the last packet and when it arrived */
static volatile uint32_t s_cur_step   = 0;
static volatile uint32_t s_last_rx_ms = 0;
static volatile bool     s_ever_rx    = false;

static void follow_task(void *arg)
{
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(15));
        if (!s_ever_rx) continue;
        uint32_t now = esp_timer_get_time() / 1000;
        if (now - s_last_rx_ms > FOLLOW_TIMEOUT_MS) {
            uint32_t nstep = s_cur_step + 1;
            uint8_t  ch    = chan_for_step(nstep);
            esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
            s_cur_step   = nstep;
            s_last_rx_ms = now;                 /* grace before the next check */
            ets_printf("HOP,RX,%u,%u,%u\n", (unsigned)(nstep & 0xFF), (unsigned)ch,
                       (unsigned)now);
        }
    }
}
#endif

static void stats_task(void *arg)
{
    uint32_t pr = 0, pc = 0;
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        uint32_t r = s_recv_count, c = s_csi_count;
        ets_printf("LINK,recv_per_s=%u,csi_per_s=%u,recv_total=%u,csi_total=%u\n",
                   (unsigned)(r - pr), (unsigned)(c - pc),
                   (unsigned)r, (unsigned)c);
        pr = r; pc = c;
    }
}

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
static void espnow_recv_cb(const esp_now_recv_info_t *info,
                           const uint8_t *data, int len)
#else
static void espnow_recv_cb(const uint8_t *mac,
                           const uint8_t *data, int len)
#endif
{
    s_recv_count++;
    if (len >= (int)sizeof(hop_pkt_t)) {
        const hop_pkt_t *p = (const hop_pkt_t *)data;
        s_last_hop_index = p->hop_index;
#if HOP_ENABLE
        s_cur_step   = p->seq / HOP_DWELL_PKTS;   /* which hop-step the TX is on */
        s_last_rx_ms = esp_timer_get_time() / 1000;
        s_ever_rx    = true;
#endif
    }
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;
    if (!info || !info->buf) return;

    const wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;
    static int s_count = 0;
    s_csi_count++;

    if (!s_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,"
                   "not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,"
                   "ampdu_cnt,channel,secondary_channel,local_timestamp,ant,"
                   "sig_len,rx_format,len,first_word,data\n");
    }
    /* metadata prefix: channel at column 16, local timestamp at column 18 */
    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
               s_count, MAC2STR(info->mac), rx->rssi, rx->rate, rx->sig_mode,
               rx->mcs, rx->cwb, rx->smoothing, rx->not_sounding,
               rx->aggregation, rx->stbc, rx->fec_coding, rx->sgi,
               rx->noise_floor, rx->ampdu_cnt, rx->channel, rx->secondary_channel,
               rx->timestamp, rx->ant, rx->sig_len, rx->sig_mode);

    ets_printf(",%d,%d,\"[%d", info->len, info->first_word_invalid, info->buf[0]);
    for (int i = 1; i < info->len; i++) {
        ets_printf(",%d", info->buf[i]);
    }
    ets_printf("]\"\n");
    s_count++;
}

static void wifi_csi_init(void)
{
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = false,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = true,
        .shift             = true,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}
#endif

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    wifi_common_init();
    espnow_common_init();

#if ROLE == ROLE_TX
    ESP_LOGW(TAG, "ROLE = TRANSMITTER");
    xTaskCreate(tx_task, "tx", 4096, NULL, 5, NULL);
#elif ROLE == ROLE_RX
    ESP_LOGW(TAG, "ROLE = RECEIVER (logging CSI)");
    ESP_ERROR_CHECK(esp_now_register_recv_cb(espnow_recv_cb));
    wifi_csi_init();
    xTaskCreate(stats_task, "stats", 3072, NULL, 4, NULL);
#if HOP_ENABLE
    s_last_rx_ms = esp_timer_get_time() / 1000;
    xTaskCreate(follow_task, "follow", 3072, NULL, 6, NULL);
    ESP_LOGW(TAG, "RX KEY-hop-follow ON: tracking 13 channels");
#endif
#else
#error "Set ROLE to ROLE_TX or ROLE_RX at the top of this file"
#endif
}
