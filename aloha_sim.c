#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int simulate_one_run(int N, double p, int T, int *remaining, int *selected, unsigned int *seed) {
    int remaining_count = N;
    for (int i = 0; i < N; i++) remaining[i] = 1;
    for (int slot = 0; slot < T; slot++) {
        int selected_count = 0;
        for (int i = 0; i < N; i++) {
            if (remaining[i]) {
                double r = (double) rand_r(seed) / RAND_MAX;
                if (r < p) selected[selected_count++] = i;
            }
        }
        if (selected_count == 1) {
            remaining[selected[0]] = 0;
            remaining_count--;
            if (remaining_count == 0) return 1;
        }
    }
    return 0;
}

int main() {
    const int N = 100;             // 端末数
    const double p = 0.01;         // 送信確率
    const int trials = 1000000;     // 試行回数（負荷対策で適度に調整）
    const int max_n = 20;          // スロット数調整（T = 100 + 100 * n）
    const char *filename = "aloha_results.csv";
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("ファイルオープンに失敗しました");
        return 1;
    }
    fprintf(fp, "T,success_rate\n");
    for (int n = 0; n <= max_n; n++) {
        int T = 100 + 100 * n;
        int success_count = 0;
        #pragma omp parallel
        {
            int remaining[N];
            int selected[N];
            unsigned int seed = (unsigned int) time(NULL) ^ omp_get_thread_num();
            #pragma omp for reduction(+:success_count)
            for (int i = 0; i < trials; i++) {
                success_count += simulate_one_run(N, p, T, remaining, selected, &seed);
            }
        }
        double success_rate = (double) success_count / trials;
        printf("T=%d, Success=%.8f\n", T, success_rate);
        fprintf(fp, "%d,%.8f\n", T, success_rate);
    }
    fclose(fp);
    printf("CSV出力完了: %s\n", filename);
    return 0;
}

