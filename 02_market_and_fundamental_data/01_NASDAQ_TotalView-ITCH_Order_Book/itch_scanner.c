/*
 * itch_scanner.c — Ultra-fast ITCH message boundary scanner
 *
 * Scans a binary ITCH buffer and extracts message offsets and types.
 * Called from Python via ctypes. Auto-compiled on first run.
 *
 * Compile: cc -O3 -shared -o itch_scanner.dylib itch_scanner.c   (macOS)
 *          cc -O3 -shared -fPIC -o itch_scanner.so itch_scanner.c (Linux)
 */

#include <stdint.h>
#include <stddef.h>

/*
 * Scan ITCH binary buffer for message boundaries.
 *
 * Each message is: [2-byte big-endian size][1-byte type][payload...]
 *
 * Returns the number of messages found (up to max_msgs).
 * Stops early if it encounters a System Event 'S' message with
 * event code 'C' (End of Messages).
 */
int scan_itch(
    const uint8_t *buf,
    size_t buf_len,
    uint64_t *offsets,   /* output: message start offsets */
    uint8_t  *types,     /* output: message type bytes */
    int max_msgs
) {
    size_t pos = 0;
    int count = 0;

    while (pos + 2 < buf_len && count < max_msgs) {
        uint16_t msg_size = ((uint16_t)buf[pos] << 8) | buf[pos + 1];
        if (msg_size == 0) break;
        if (pos + 2 + msg_size > buf_len) break;

        offsets[count] = (uint64_t)pos;
        types[count] = buf[pos + 2];
        count++;

        /* Check for System Event 'S' (0x53) with End of Messages 'C' (0x43) */
        if (buf[pos + 2] == 0x53) {
            /* event_code is at a fixed offset within the 'S' message:
             * stock_locate(2) + tracking_number(2) + timestamp(6) + event_code(1)
             * = offset pos+3 + 10 = pos+13 */
            if (pos + 13 < buf_len && buf[pos + 13] == 0x43) {
                break;
            }
        }

        pos += 2 + msg_size;
    }

    return count;
}
