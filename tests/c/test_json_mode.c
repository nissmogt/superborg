/* JSON recognizer tests: accepts exactly the texts json.loads would (for single values). */
#include "internal.h"

#include <stdio.h>
#include <string.h>

static int accepts(const char *text, int *complete) {
    json_state s;
    json_state_init(&s);
    for (const char *p = text; *p; p++)
        if (!json_state_feed(&s, (unsigned char)*p)) return 0;
    /* a top-level number is complete once followed by whitespace */
    json_state t = s;
    *complete = json_state_feed(&t, ' ') && (t.mode == 9 /* J_DONE */);
    return 1;
}

int main(void) {
    const char *good[] = {"{}", "[]", "{\"a\": 1}", "[1, 2.5, -3e+4, 0, -0.0, 1E9]", "\"x\\n\\u00e9\\\"\"",
                          "{\"a\": {\"b\": [true, false, null]}, \"c\": \"d\"}", "  [ ]  ", "42", "-0.5e-3",
                          "{\"k\": \"caf\xc3\xa9\"}", "[[[[]]]]", "\"\""};
    const char *bad[] = {"{,}", "[1,]", "{\"a\" 1}", "{a: 1}", "[01]", "[1.]", "[.5]", "tru", "[True]", "{\"a\":1}}",
                         "[1 2]", "\"\x01\"", "\"\\x\"", "[-]", "{\"a\":}", "]"};
    int fails = 0, complete;
    for (size_t i = 0; i < sizeof good / sizeof *good; i++)
        if (!accepts(good[i], &complete) || !complete) { printf("FAIL good: %s\n", good[i]); fails++; }
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        int ok = accepts(bad[i], &complete);
        if (ok && complete) { printf("FAIL bad accepted: %s\n", bad[i]); fails++; }
    }
    /* prefixes of valid texts must be accepted but not complete */
    if (!accepts("{\"a\": [1, 2", &complete) || complete) { printf("FAIL prefix\n"); fails++; }
    printf(fails ? "%d FAILURES\n" : "all json mode tests passed\n", fails);
    return fails != 0;
}
