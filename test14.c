#include <stdio.h>
#include <string.h>
#include "esp_wifi.h"
#include "esp_system.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include <stdint.h>
// Wi-Fiフレームヘッダー構造体定義
typedef struct {
    unsigned frame_ctrl:16;
    unsigned duration_id:16;
    unsigned sequence_ctrl:16;
    uint8_t addr1[6];  // 宛先MACアドレス
    uint8_t addr2[6];  // 送信元MACアドレス
    uint8_t addr3[6];  // フィルタリングアドレス
    uint8_t addr4[6];  // オプション（アドホックなどの場合）
} wifi_ieee80211_mac_hdr_t;
typedef struct {
    wifi_ieee80211_mac_hdr_t hdr;
    uint8_t payload[0];  // フレームデータ部分
} wifi_ieee80211_packet_t;
// ブラックリストに登録されたMACアドレスの先頭3バイト（OUI）
const uint8_t mac_blacklist[][6] = {
    //{0xD4, 0x2C, 0x46},
    //{0x10, 0x6f, 0x3f}
    {0x78, 0x4f, 0x43, 0x97, 0x8e, 0x32}
};
const size_t mac_blacklist_size = sizeof(mac_blacklist) / sizeof(mac_blacklist[0]);
// MACアドレスを表示する関数
void print_mac(const uint8_t *mac) {
    printf("%02x:%02x:%02x:%02x:%02x:%02x",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}
// MACアドレスの妥当性を確認する関数
int is_valid_mac(const uint8_t *mac) {
    for (int i = 0; i < 6; i++) {
        if (mac[i] != 0x00 && mac[i] != 0xFF) {
            return 0;  // 有効
        }
    }
    return 1;  // 無効
}
// ブラックリストに一致するか確認する関数
int is_mac_blacklisted(const uint8_t *mac) {
    for (size_t i = 0; i < mac_blacklist_size; i++) {
        for (size_t i = 0; i < mac_blacklist_size; i++) {
            if (memcmp(mac, mac_blacklist[i], 6) == 0) {
                return 0;
            }
        }
    }
    return 1; // 一致しない
}
// パケットの種類を分類して文字列を返す関数
const char* classify_packet(unsigned frame_type, unsigned frame_subtype) {
    if (frame_type == 0) { // 管理フレーム
        switch (frame_subtype) {
            case 8: return "Beacon";
            case 4: return "Probe Request";
            case 5: return "Probe Response";
            case 0: return "Association Request";
            case 1: return "Association Response";
            case 2: return "Reassociation Request";
            case 3: return "Reassociation Response";
            case 9: return "ATIM";
            case 10: return "Disassociation";
            case 11: return "Authentication";
            case 12: return "Deauthentication";
            default: return "Unknown Management";
        }
    } else if (frame_type == 1) { // 制御フレーム
        switch (frame_subtype) {
            case 7: return "Control Wrapper";
            case 8: return "Block ACK Request";
            case 9: return "Block ACK";
            case 10: return "Power Save Poll";
            case 11: return "RTS";
            case 12: return "CTS";
            case 13: return "ACK";
            case 14: return "CF-End";
            case 15: return "CF-End + CF-Ack";
            default: return "Unknown Control";
        }
    } else if (frame_type == 2) { // データフレーム
        return "Data";
    } else {
        return "Unknown Frame Type";
    }
}
// パケットの正常性を確認する関数
int is_packet_valid(const wifi_ieee80211_mac_hdr_t *hdr, uint16_t length) {
    // Frame Control解析
    unsigned protocol_version = hdr->frame_ctrl & 0x0003;
    unsigned frame_type = (hdr->frame_ctrl & 0x000C) >> 2;
    unsigned frame_subtype = (hdr->frame_ctrl & 0x00F0) >> 4;
    if (protocol_version != 0) {
        printf("Invalid protocol version: %u\n", protocol_version);
        return 0;  // 異常パケット
    }
    if (frame_type > 2) {
        printf("Unknown frame type: %u\n", frame_type);
        return 0;  // 異常パケット
    }
    // MACアドレスの妥当性
//    if (!is_valid_mac(hdr->addr1) || !is_valid_mac(hdr->addr2)) {
//        printf("Invalid MAC address detected.\n");
//        return 0;  // 異常パケット
//    }
    // パケット長の確認
    if (length < sizeof(wifi_ieee80211_mac_hdr_t)) {
        printf("Packet too short: %u bytes\n", length);
        return 0;  // 異常パケット
    }
    return 1;  // 正常パケット
}
// キャリブレーション関数
void calibrate_channel(void) {
    printf("Starting channel calibration...\n");
    for (int channel = 1; channel <= 13; channel++) {
        printf("Testing channel %d...\n", channel);
        esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE);
        // 一定期間パケットをキャプチャ
        vTaskDelay(pdMS_TO_TICKS(500));  // 500ms待機
        // チャネルごとのパケット数をログ
        printf("Channel %d calibration complete.\n", channel);
    }
    printf("Channel calibration complete.\n");
}
// パケットのダンプ関数
void dump_packet(const uint8_t *data, uint16_t length) {
    printf("Packet Dump: ");
    for (int i = 0; i < length; i++) {
        printf("%02x ", data[i]);
    }
    printf("\n");
}
// フレームコントロールの解析関数
void parse_frame_control(unsigned frame_ctrl) {
    unsigned protocol_version = frame_ctrl & 0x0003;
    unsigned frame_type = (frame_ctrl & 0x000C) >> 2;
    unsigned frame_subtype = (frame_ctrl & 0x00F0) >> 4;
    printf("Frame Control Analysis:\n");
    printf("  Protocol Version: %u\n", protocol_version);
    printf("  Frame Type: %u\n", frame_type);
    printf("  Frame Subtype: %u\n", frame_subtype);
}
// Wi-Fiパケットを処理するコールバック関数
void wifi_sniffer_packet_handler(void *buff, wifi_promiscuous_pkt_type_t type) {
    const wifi_promiscuous_pkt_t *pkt = (wifi_promiscuous_pkt_t *)buff;
    const uint8_t *raw_data = pkt->payload;
    dump_packet(raw_data, pkt->rx_ctrl.sig_len);
    const wifi_ieee80211_packet_t *ipkt = (wifi_ieee80211_packet_t *)raw_data;
    const wifi_ieee80211_mac_hdr_t *hdr = &ipkt->hdr;
    unsigned header_size = sizeof(wifi_ieee80211_mac_hdr_t);
    printf("Header size: %u bytes\n", header_size);
    parse_frame_control(hdr->frame_ctrl);
    if (!is_packet_valid(hdr, pkt->rx_ctrl.sig_len)) {
        printf("Invalid packet detected, skipping...\n");
        return;
    }
    // ブラックリストに一致する場合は表示しない
    if (is_mac_blacklisted(hdr->addr2)) {
        return;
    }
    printf("\n--- Wi-Fi Packet Captured ---\n");
    // MACアドレスを表示
    printf("Destination MAC: ");
    print_mac(hdr->addr1);
    printf("\n");
    printf("Source MAC: ");
    print_mac(hdr->addr2);
    printf("\n");
    printf("BSSID: ");
    print_mac(hdr->addr3);
    printf("\n");
    // フレームタイプとサブタイプを抽出
    unsigned frame_type = (hdr->frame_ctrl & 0x000C) >> 2;
    unsigned frame_subtype = (hdr->frame_ctrl & 0x00F0) >> 4;
    // パケットの種類を分類して表示
    const char* packet_type = classify_packet(frame_type, frame_subtype);
    printf("Packet Type: %s\n", packet_type);
}
// アプリケーションのエントリーポイント
void app_main(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    // キャリブレーション実行
    calibrate_channel();
    // Wi-FiをNULLモードに設定（プロミスキャスモード専用）
    esp_wifi_set_mode(WIFI_MODE_NULL);
    esp_wifi_start();
    wifi_promiscuous_filter_t filter = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_ALL // すべてのフレームを許可
    };
    esp_wifi_set_promiscuous_filter(&filter);
    // プロミスキャスモードを有効化
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_promiscuous_rx_cb(wifi_sniffer_packet_handler);
    printf("Wi-Fi Sniffer Initialized.\n");
}
