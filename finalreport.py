import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
import math
import argparse

@dataclass
class LinkParams:
    N_nodes: int = 100
    data_bits_per_node: int = 56
    distance_m: float = 400.0
    wavelength_m: float = 0.1224
    pathloss_exp: float = 2.6
    EIRP_dBm: float = 0.0
    Gt_dBi: float = 0.0
    Gr_dBi: float = 5.4
    T_K: float = 927.1
    k_B: float = 1.380649e-23
    duration_s: float = 60.0
    n_slots: int = 1800
    rolloff_alpha: float = 0.25

@dataclass
class AlohaParams:
    p: float = 0.01
    allow_capture: bool = False

@dataclass
class CodeParams:
    n: int = 7
    k: int = 4

def dbm_to_watt(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)

def watt_to_dbm(w: float) -> float:
    return 10 * math.log10(w) + 30.0

def lin_to_db(x: float) -> float:
    return 10.0 * math.log10(x)

class HammingCode:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        r = n - k
        if n != (1 << r) - 1:
            raise ValueError("Hamming: n must be 2^r - 1.")
        self.r = r
        self.parity_pos = [1 << i for i in range(r)]         # 1,2,4,...
        self.data_pos = [i for i in range(1, n + 1) if i not in self.parity_pos]
        if len(self.data_pos) != k:
            raise ValueError("Hamming: k mismatch.")
        self.codebook_bits = np.zeros((1 << k, n), dtype=np.int8)
        self.codebook_pm = np.zeros((1 << k, n), dtype=np.int8)  # 0->+1,1->-1
        for msg in range(1 << k):
            bits = self._encode_bits(int_to_bits(msg, k))
            self.codebook_bits[msg, :] = bits
            self.codebook_pm[msg, :] = (1 - 2 * bits)

    def _encode_bits(self, data_bits: np.ndarray) -> np.ndarray:
        cw = np.zeros(self.n, dtype=np.int8)
        for i, pos in enumerate(self.data_pos):
            cw[pos - 1] = data_bits[i]
        for i, ppos in enumerate(self.parity_pos):
            parity = 0
            for bit_index in range(1, self.n + 1):
                if bit_index & ppos:
                    parity ^= cw[bit_index - 1]
            cw[ppos - 1] = parity
        return cw

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        assert data_bits.ndim == 1
        assert data_bits.size % self.k == 0
        m = data_bits.size // self.k
        out = np.zeros(m * self.n, dtype=np.int8)
        for i in range(m):
            blk = data_bits[i * self.k:(i + 1) * self.k]
            cw = self._encode_bits(blk)
            out[i * self.n:(i + 1) * self.n] = cw
        return out

    def decode_soft_llr(self, llr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert llr.size % self.n == 0
        m = llr.size // self.n
        data_out = np.zeros(m * self.k, dtype=np.int8)
        code_out = np.zeros(m * self.n, dtype=np.int8)
        for i in range(m):
            block_llr = llr[i * self.n:(i + 1) * self.n]
            metrics = self.codebook_pm @ block_llr
            best = int(np.argmax(metrics))
            best_cw = self.codebook_bits[best]
            code_out[i * self.n:(i + 1) * self.n] = best_cw
            for j, pos in enumerate(self.data_pos):
                data_out[i * self.k + j] = best_cw[pos - 1]
        return data_out, code_out

def int_to_bits(x: int, nbits: int) -> np.ndarray:
    return np.array([(x >> i) & 1 for i in range(nbits)], dtype=np.int8)[::-1]

def qpsk_mod(bits: np.ndarray, Eb: float) -> np.ndarray:
    assert bits.size % 2 == 0
    b = bits.reshape(-1, 2)
    sI = 1 - 2 * b[:, 0]
    sQ = 1 - 2 * b[:, 1]
    amp = math.sqrt(Eb)   # 各軸 ±sqrt(Eb) → Es = 2Eb
    return amp * (sI.astype(float) + 1j * sQ.astype(float))

def qpsk_awgn(sym: np.ndarray, N0: float, rng: np.random.Generator) -> np.ndarray:
    sigma = math.sqrt(N0 / 2.0)
    noise = sigma * (rng.normal(size=sym.shape) + 1j * rng.normal(size=sym.shape))
    return sym + noise

def qpsk_llr(sym: np.ndarray, Eb: float, N0: float) -> np.ndarray:
    kappa = 2.0 * math.sqrt(Eb) / N0
    llr_I = kappa * sym.real
    llr_Q = kappa * sym.imag
    out = np.empty(sym.size * 2, dtype=float)
    out[0::2] = llr_I
    out[1::2] = llr_Q
    return out

def compute_Pr_dBm(cfg: LinkParams) -> float:
    base = (4.0 * math.pi * cfg.distance_m) / cfg.wavelength_m
    L_lin = base ** cfg.pathloss_exp
    L_dB = lin_to_db(L_lin)
    Pr_dBm = cfg.EIRP_dBm - L_dB + cfg.Gr_dBi
    return Pr_dBm

def slot_and_rates(cfg: LinkParams, code: HammingCode) -> Dict[str, float]:
    slot_duration = cfg.duration_s / cfg.n_slots
    n_blocks = math.ceil(cfg.data_bits_per_node / code.k)
    coded_bits = n_blocks * code.n
    Rb_coded = coded_bits / slot_duration
    Rb_true = cfg.data_bits_per_node / slot_duration  # 非符号化想定
    return dict(slot_duration=slot_duration,
                n_blocks=n_blocks,
                coded_bits=coded_bits,
                Rb_coded=Rb_coded,
                Rb_true=Rb_true)

@dataclass
class SimResult:
    success_nodes_avg: float
    success_fraction: float
    throughput_bps: float
    ber_coded: float
    fer_coded: float
    ber_uncoded_tx_vs_dec_code: float
    ber_uncoded_true: float
    trials: int

def simulate(cfg: LinkParams,
             aloha: AlohaParams,
             code_params: CodeParams,
             trials: int,
             seed: int = 1) -> SimResult:
    rng = np.random.default_rng(seed)
    code = HammingCode(code_params.n, code_params.k)
    Pr_dBm = compute_Pr_dBm(cfg)
    Pr_W = dbm_to_watt(Pr_dBm)
    rates = slot_and_rates(cfg, code)
    slot_duration = rates["slot_duration"]
    coded_bits = int(rates["coded_bits"])
    Rb_coded = rates["Rb_coded"]
    Rb_true  = rates["Rb_true"]
    N0 = cfg.k_B * cfg.T_K
    Eb_coded = Pr_W / Rb_coded
    Eb_true  = Pr_W / Rb_true
    def ber_qpsk_theory(Eb, N0):
        return 0.5 * math.erfc(math.sqrt(Eb / N0))
    ber_th_coded = ber_qpsk_theory(Eb_coded, N0)
    ber_th_true  = ber_qpsk_theory(Eb_true,  N0)
    total_info_bits = 0
    total_bit_err_coded = 0
    total_frame_err_coded = 0
    total_coded_bits = 0
    total_err_tx_vs_dec_code = 0
    total_info_bits_true = 0
    total_err_uncoded_true = 0
    success_nodes_sum = 0
    N = cfg.N_nodes
    T = cfg.n_slots
    p = aloha.p
    for _ in range(trials):
        pending = np.ones(N, dtype=bool)
        success_nodes = 0
        info_all = [rng.integers(0, 2, size=cfg.data_bits_per_node, dtype=np.int8)
                    for _ in range(N)]
        for _slot in range(T):
            to_send = np.where(pending & (rng.random(N) < p))[0]
            if to_send.size == 1:
                nidx = to_send[0]
                info_bits = info_all[nidx]
                pad_len = ((math.ceil(info_bits.size / code.k) * code.k) - info_bits.size)
                info_padded = info_bits if pad_len == 0 else np.concatenate(
                    [info_bits, np.zeros(pad_len, dtype=np.int8)]
                )
                coded = code.encode(info_padded)
                tx_coded = qpsk_mod(coded, Eb_coded)
                rx_coded = qpsk_awgn(tx_coded, N0, rng)
                llr_coded = qpsk_llr(rx_coded, Eb_coded, N0)
                dec_data, dec_code = code.decode_soft_llr(llr_coded)
                used_dec_data = dec_data[:cfg.data_bits_per_node]
                bit_err_coded = np.count_nonzero(used_dec_data != info_bits)
                total_bit_err_coded += bit_err_coded
                total_frame_err_coded += (1 if bit_err_coded > 0 else 0)
                total_info_bits += info_bits.size
                dec_code_hard = (llr_coded < 0).astype(np.int8)
                err_tx_vs_dec = np.count_nonzero(dec_code_hard != coded)
                total_err_tx_vs_dec_code += err_tx_vs_dec
                total_coded_bits += coded.size
                pad2 = info_bits.size % 2
                info_true = info_bits if pad2 == 0 else np.concatenate(
                    [info_bits, np.zeros(1, dtype=np.int8)]
                )
                tx_true = qpsk_mod(info_true, Eb_true)
                rx_true = qpsk_awgn(tx_true, N0, rng)
                llr_true = qpsk_llr(rx_true, Eb_true, N0)
                dec_true_hard = (llr_true[:info_bits.size] < 0).astype(np.int8)
                err_true = np.count_nonzero(dec_true_hard != info_bits)
                total_err_uncoded_true += err_true
                total_info_bits_true += info_bits.size
                pending[nidx] = False
                success_nodes += 1
                if success_nodes == N:
                    break
            else:
                pass
        success_nodes_sum += success_nodes
    success_nodes_avg = success_nodes_sum / trials
    success_fraction = success_nodes_avg / N
    throughput_bps = (success_nodes_avg * cfg.data_bits_per_node) / cfg.duration_s
    ber_coded = total_bit_err_coded / max(1, total_info_bits)
    fer_coded = total_frame_err_coded / max(1, trials * N)
    ber_uncoded_tx_vs_dec_code = total_err_tx_vs_dec_code / max(1, total_coded_bits)
    ber_uncoded_true = total_err_uncoded_true / max(1, total_info_bits_true)
    print("==== Link & Rate ====")
    print(f"Pr_dBm                : {Pr_dBm:.2f} dBm")
    print(f"N0                    : {N0:.3e} W/Hz")
    print(f"slot_duration         : {slot_duration:.6f} s")
    print(f"coded_bits_per_frame  : {coded_bits}")
    print(f"Rb_coded              : {Rb_coded:.3f} bit/s")
    print(f"Rb_true  (no coding)  : {Rb_true:.3f} bit/s")
    print(f"Eb_coded              : {Eb_coded:.3e} J/bit")
    print(f"Eb_true               : {Eb_true:.3e} J/bit")
    print(f"BER_QPSK_theory_coded : {ber_th_coded:.3e}")
    print(f"BER_QPSK_theory_true  : {ber_th_true:.3e}")
    print()
    print("==== Traffic (ALOHA) ====")
    print(f"p                      : {aloha.p}")
    print(f"success_nodes_avg      : {success_nodes_avg:.3f} / {N}")
    print(f"success_fraction       : {success_fraction:.6f}")
    print(f"throughput             : {throughput_bps:.6f} bit/s")
    print()
    print("==== Errors ====")
    print(f"BER_coded (soft-dec)           : {ber_coded:.6e}")
    print(f"FER_coded (soft-dec)           : {fer_coded:.6e}")
    print(f"BER_uncoded_tx_vs_dec_code     : {ber_uncoded_tx_vs_dec_code:.6e}  # coded vs hard判定")
    print(f"BER_uncoded_true (no-coding)   : {ber_uncoded_true:.6e}  # 情報56bitを直接送信")
    return SimResult(
        success_nodes_avg=success_nodes_avg,
        success_fraction=success_fraction,
        throughput_bps=throughput_bps,
        ber_coded=ber_coded,
        fer_coded=fer_coded,
        ber_uncoded_tx_vs_dec_code=ber_uncoded_tx_vs_dec_code,
        ber_uncoded_true=ber_uncoded_true,
        trials=trials,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--code", type=str, default="7,4")   # or 15,11
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    n_s, k_s = args.code.split(",")
    code_params = CodeParams(n=int(n_s), k=int(k_s))
    cfg = LinkParams()
    aloha = AlohaParams(p=args.p)
    simulate(cfg, aloha, code_params, trials=args.trials, seed=args.seed)

if __name__ == "__main__":
    main()
