"""
TSM — Kamera Bazlı Tablo Bölme Scripti
=======================================

Mevcut TSM sequence dosyalarından her kamera için ayrı flat tablolar üretir.
SlowFast versiyonuyla aynı mantık — yalnızca dosya isimleri ve feature
boyutu farklıdır (TSM: 4064 dim, SlowFast: 6336 dim).

Girdi dosyaları (mevcut):
    train_features_correct_tsm_sequences.npy
    train_features_correct_tsm_sequence_ids.npy
    test_sequences_tsm.npy
    test_sequence_ids_tsm.npy
    test_sequence_window_labels_tsm.npy

Çıktı dosyaları (kamera bazlı):
    train_features_correct_tsm_C10095.npy, ...  (6 kamera)
    test_features_tsm_C10095.npy, ...           (6 kamera)
    test_window_labels_tsm_C10095.npy, ...      (6 kamera)

Yol B (per-camera Autoencoder) için:
    Her kameranın train dosyasıyla ayrı bir Autoencoder eğitilir.
    Test sırasında 6 kameradan 6 karar alınır, çoğunluk oylaması uygulanır.
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------

BASE_DIR = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection"

CAMERAS = ['C10095', 'C10115', 'C10118', 'C10119', 'C10390', 'C10404']

TRAIN_SEQ_FILE = os.path.join(BASE_DIR, "train_features_correct_tsm_sequences.npy")
TRAIN_ID_FILE  = os.path.join(BASE_DIR, "train_features_correct_tsm_sequence_ids.npy")
TEST_SEQ_FILE  = os.path.join(BASE_DIR, "test_sequences_tsm.npy")
TEST_ID_FILE   = os.path.join(BASE_DIR, "test_sequence_ids_tsm.npy")
TEST_WLBL_FILE = os.path.join(BASE_DIR, "test_sequence_window_labels_tsm.npy")

OUT_DIR = os.path.join(BASE_DIR, "per_camera_tsm")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------

def load_files(seq_file, id_file, wlbl_file=None):
    print(f"  Yükleniyor: {os.path.basename(seq_file)}")
    sequences = np.load(seq_file, allow_pickle=True)
    seq_ids   = np.load(id_file,  allow_pickle=True)
    window_labels = None
    if wlbl_file and os.path.exists(wlbl_file):
        print(f"  Yükleniyor: {os.path.basename(wlbl_file)}")
        window_labels = np.load(wlbl_file, allow_pickle=True)
    print(f"  Toplam sekans: {len(sequences)}")
    return sequences, seq_ids, window_labels


def split_by_camera(sequences, seq_ids, window_labels=None):
    camera_data = {cam: [] for cam in CAMERAS}
    camera_wlbl = {cam: [] for cam in CAMERAS}
    unmatched   = []

    for i, (seq, sid) in enumerate(zip(sequences, seq_ids)):
        matched = False
        for cam in CAMERAS:
            if cam in sid:
                camera_data[cam].append(seq)
                if window_labels is not None:
                    camera_wlbl[cam].append(window_labels[i])
                matched = True
                break
        if not matched:
            unmatched.append(sid)

    if unmatched:
        print(f"  UYARI: {len(unmatched)} sekans eşleşmedi: {unmatched}")

    return camera_data, camera_wlbl


def save_camera_tables(camera_data, camera_wlbl, prefix, is_test=False):
    print(f"\n  Kaydediliyor — prefix: {prefix}")
    summary = []

    for cam in CAMERAS:
        seqs = camera_data[cam]
        if not seqs:
            print(f"  UYARI: {cam} için hiç sekans yok — atlanıyor")
            continue

        flat = np.vstack(seqs)
        feat_path = os.path.join(OUT_DIR, f"{prefix}_{cam}.npy")
        np.save(feat_path, flat)

        if is_test and camera_wlbl[cam]:
            wlbl     = np.concatenate(camera_wlbl[cam])
            lbl_path = os.path.join(OUT_DIR, f"{prefix}_window_labels_{cam}.npy")
            np.save(lbl_path, wlbl)
            n_correct = (wlbl == 0).sum()
            n_anomaly = (wlbl == 1).sum()
            print(f"  ✅ {cam}: {flat.shape} | correct:{n_correct} anomaly:{n_anomaly}")
            summary.append((cam, flat.shape[0], n_correct, n_anomaly))
        else:
            print(f"  ✅ {cam}: {flat.shape}")
            summary.append((cam, flat.shape[0], flat.shape[0], 0))

    return summary


def validate_split(camera_data, original_total):
    split_total = sum(
        sum(len(seq) for seq in seqs)
        for seqs in camera_data.values()
    )
    ok = split_total == original_total
    print(f"  {'✅' if ok else '❌'} Toplam pencere: orijinal={original_total} bölünmüş={split_total}")


# ---------------------------------------------------------------------------
# Ana İşlem
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  TSM — Kamera Bazlı Tablo Bölme (6 Kamera)")
    print("=" * 60)
    print(f"  Kameralar: {CAMERAS}")
    print(f"  Feature boyutu: 4064 dim (visual 2048 + pose 2016)")

    # --- Eğitim ---
    print("\n[1/2] EĞİTİM VERİSİ")
    print("-" * 40)

    if not os.path.exists(TRAIN_SEQ_FILE):
        print(f"❌ Bulunamadı: {TRAIN_SEQ_FILE}")
        print(f"   TSM extraction tamamlandı mı kontrol et.")
        return

    train_seqs, train_ids, _ = load_files(TRAIN_SEQ_FILE, TRAIN_ID_FILE)
    original_train = sum(len(s) for s in train_seqs)
    print(f"  Orijinal toplam pencere: {original_train}")

    train_cam_data, _ = split_by_camera(train_seqs, train_ids)

    print("\n  Kamera bazında dağılım:")
    for cam in CAMERAS:
        n_seqs    = len(train_cam_data[cam])
        n_windows = sum(len(s) for s in train_cam_data[cam])
        print(f"    {cam}: {n_seqs} sekans, {n_windows} pencere")

    validate_split(train_cam_data, original_train)
    train_summary = save_camera_tables(
        train_cam_data, {}, prefix="train_features_correct_tsm", is_test=False)

    # --- Test ---
    print("\n[2/2] TEST VERİSİ")
    print("-" * 40)

    if not os.path.exists(TEST_SEQ_FILE):
        print(f"❌ Bulunamadı: {TEST_SEQ_FILE}")
        print(f"   TSM test extraction tamamlandı mı kontrol et.")
        return

    test_seqs, test_ids, test_wlbls = load_files(
        TEST_SEQ_FILE, TEST_ID_FILE, TEST_WLBL_FILE)
    original_test = sum(len(s) for s in test_seqs)
    print(f"  Orijinal toplam pencere: {original_test}")

    test_cam_data, test_cam_wlbl = split_by_camera(
        test_seqs, test_ids, test_wlbls)

    print("\n  Kamera bazında dağılım:")
    for cam in CAMERAS:
        n_seqs    = len(test_cam_data[cam])
        n_windows = sum(len(s) for s in test_cam_data[cam])
        print(f"    {cam}: {n_seqs} sekans, {n_windows} pencere")

    validate_split(test_cam_data, original_test)
    test_summary = save_camera_tables(
        test_cam_data, test_cam_wlbl, prefix="test_features_tsm", is_test=True)

    # --- Özet ---
    print("\n" + "=" * 60)
    print("  ÖZET")
    print("=" * 60)
    print(f"\n  Çıktı klasörü: {OUT_DIR}")
    print(f"\n  Eğitim dosyaları (TSM — 4064 dim):")
    for cam, n_win, _, _ in train_summary:
        print(f"    train_features_correct_tsm_{cam}.npy → ({n_win}, 4064)")
    print(f"\n  Test dosyaları (TSM — 4064 dim):")
    for cam, n_win, n_c, n_a in test_summary:
        print(f"    test_features_tsm_{cam}.npy → ({n_win}, 4064) "
              f"[correct:{n_c} anomaly:{n_a}]")
    print(f"\n  Yol B kullanımı:")
    print(f"    Her kamera için ayrı Autoencoder eğit → 6 model")
    print(f"    Test: 6 karardan 4'ü anomali → ANOMALİ")
    print(f"          6 karardan 4'ü normal  → NORMAL")
    print("\n✅ Tamamlandı.")


if __name__ == "__main__":
    main()