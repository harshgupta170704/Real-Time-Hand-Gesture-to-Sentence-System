"""
Gesture trainer module.
Collects hand landmark data for each gesture and trains a Random Forest classifier.
"""

import os
import csv
import pickle
import numpy as np
import cv2
import config


def collect_training_data(camera, hand_detector):
    """
    Interactive data collection for gesture training.
    Shows camera feed and lets user record landmark data for each gesture.

    Args:
        camera: Camera instance.
        hand_detector: HandDetector instance.
    """
    os.makedirs(config.TRAINING_DATA_DIR, exist_ok=True)

    gestures = list(config.GESTURE_WORD_MAP.keys())
    print("\n" + "=" * 60)
    print("  GESTURE TRAINING DATA COLLECTION")
    print("=" * 60)
    print(f"\nYou will record {config.TRAINING_SAMPLES_PER_GESTURE} samples")
    print(f"for each of the {len(gestures)} gestures.\n")

    for i, gesture in enumerate(gestures):
        word = config.GESTURE_WORD_MAP[gesture]
        filepath = os.path.join(config.TRAINING_DATA_DIR, f"{gesture}.csv")

        print(f"\n--- Gesture {i+1}/{len(gestures)}: '{gesture}' → '{word}' ---")
        print("Hold the gesture steady in front of the camera.")
        print("Press 'r' to start recording, 'n' to skip, 'q' to quit.\n")

        samples = []
        recording = False
        sample_count = 0

        while True:
            frame_rgb = camera.read()
            if frame_rgb is None:
                continue

            result = hand_detector.detect(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            if config.SHOW_LANDMARKS:
                hand_detector.draw_landmarks(frame_bgr, result["raw_results"])

            # Status overlay
            status_text = f"Gesture: {gesture} ({word})"
            cv2.putText(frame_bgr, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_PRIMARY, 2)

            if recording:
                progress = f"Recording: {sample_count}/{config.TRAINING_SAMPLES_PER_GESTURE}"
                cv2.putText(frame_bgr, progress, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_SUCCESS, 2)

                # Collect sample if hand is detected
                if result["normalized"] is not None:
                    samples.append(result["normalized"])
                    sample_count += 1

                    if sample_count >= config.TRAINING_SAMPLES_PER_GESTURE:
                        recording = False
                        print(f"  ✓ Collected {sample_count} samples for '{gesture}'")
            else:
                hint = "Press 'r' to record, 'n' to skip, 'q' to quit"
                cv2.putText(frame_bgr, hint, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_TEXT, 1)

            cv2.imshow("Gesture Trainer", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r') and not recording:
                recording = True
                samples = []
                sample_count = 0
                print(f"  Recording started for '{gesture}'...")
            elif key == ord('n'):
                print(f"  Skipped '{gesture}'")
                break
            elif key == ord('q'):
                print("  Training cancelled.")
                cv2.destroyAllWindows()
                return

            if sample_count >= config.TRAINING_SAMPLES_PER_GESTURE:
                # Save to CSV
                _save_samples(filepath, gesture, samples)
                break

    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("  DATA COLLECTION COMPLETE!")
    print(f"  Data saved in '{config.TRAINING_DATA_DIR}/'")
    print("=" * 60)


def _save_samples(filepath, gesture_name, samples):
    """Save landmark samples to a CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
        writer.writerow(header)
        for sample in samples:
            writer.writerow([gesture_name] + sample)
    print(f"  Saved {len(samples)} samples to {filepath}")


def train_model():
    """
    Train a Random Forest classifier on collected gesture data.

    Returns:
        dict: Training results with accuracy info.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report

    print("\n" + "=" * 60)
    print("  TRAINING GESTURE CLASSIFIER")
    print("=" * 60)

    # Load all CSV files
    all_features = []
    all_labels = []

    if not os.path.exists(config.TRAINING_DATA_DIR):
        print(f"ERROR: No training data found in '{config.TRAINING_DATA_DIR}/'")
        print("Run data collection first!")
        return None

    csv_files = [f for f in os.listdir(config.TRAINING_DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("ERROR: No CSV files found in training data directory.")
        return None

    for csv_file in csv_files:
        filepath = os.path.join(config.TRAINING_DATA_DIR, csv_file)
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                label = row[0]
                features = [float(x) for x in row[1:]]
                all_labels.append(label)
                all_features.append(features)

    print(f"Loaded {len(all_features)} samples across {len(csv_files)} gestures.")

    X = np.array(all_features)
    y = np.array(all_labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train Random Forest
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))

    # Save model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_data = {
        "model": model,
        "label_encoder": label_encoder,
    }
    with open(config.ML_MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to '{config.ML_MODEL_PATH}'")

    return {
        "accuracy": accuracy,
        "num_samples": len(all_features),
        "num_gestures": len(csv_files),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gesture Training Tool")
    parser.add_argument(
        "--mode", choices=["collect", "train", "both"],
        default="both", help="Mode: collect data, train model, or both."
    )
    parser.add_argument(
        "--camera", choices=["picamera", "webcam"],
        default=config.CAMERA_SOURCE, help="Camera source."
    )
    args = parser.parse_args()

    if args.mode in ("collect", "both"):
        from camera import Camera
        from hand_detector import HandDetector

        cam = Camera(source=args.camera)
        detector = HandDetector()
        cam.start()
        try:
            collect_training_data(cam, detector)
        finally:
            cam.stop()
            detector.close()

    if args.mode in ("train", "both"):
        train_model()
