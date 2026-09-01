#Overview
The proposed method is a lightweight NLoS fire and human detection system based on Wi-Fi Channel State Information (CSI) for early fire warning in high-rise buildings. It uses Wi-Fi signals that can propagate through obstacles to sense fire- and human-induced channel variations in corridor, through-wall, and cross-floor scenarios.
An ESP32 collects CSI amplitude and phase data and sends them to an STM32 through UART. The STM32 performs phase sanitization, moving-average filtering, subcarrier selection, feature normalization, and lightweight MLP classification for four states: NFN, NFH, FHN, and FWH.
#Highlights
•	Through-Obstacle Sensing: Supports corridor, through-wall, and cross-floor NLoS fire detection.
•	Amplitude + Phase Fusion: Uses both CSI amplitude and sanitized phase information.
•	MCU-Friendly: Reduces CSI feature dimensionality for real-time STM32 deployment.
•	Fire + Human Recognition: Distinguishes NFN, NFH, FHN, and FWH.
•	Strong Performance: Achieves 96.12% accuracy at the recommended 10 Hz sampling rate.
#Method
•	Phase Sanitization: Removes phase distortions caused by hardware and synchronization offsets.
•	Moving-Average Filtering: Suppresses short-term noise with a filtering-window length of 10.
•	SNR-Based Amplitude Selection: Retains strong and stable amplitude subcarriers.
•	Fisher-Based Phase Selection: Retains phase subcarriers with strong class separability.
•	Feature Fusion + MLP: Normalized amplitude and phase features are concatenated and classified by a lightweight MLP with Softmax output.
#Experimental Setup
•	Platform: ESP32 + STM32 connected via UART.
•	Scenarios: Corridor, through-wall, and cross-floor NLoS environments.
•	Classes: NFN, NFH, FHN, and FWH.
•	Optimal Features: 8 amplitude and 12 phase subcarriers, reducing the original 104-dimensional representation to 20 dimensions.
•	Ablation Result: At 100 Hz, amplitude + phase achieves 97.42% accuracy; removing amplitude gives 94.28%, while removing phase gives 93.52%.
•	Recommended Sampling Rate: 10 Hz achieves 96.12% accuracy with low RAM, Flash, and inference-latency requirements.
Advantages
•	NLoS Coverage beyond traditional LoS sensing.
•	Low Cost by reusing commodity Wi-Fi infrastructure.
•	Privacy-Preserving because no visual information is required.
•	Human-Aware Detection reduces confusion between fire-induced and human-induced signal variations.
•	Lightweight Deployment supports real-time inference on STM32.
#System Architecture
•	Acquisition: ESP32 collects CSI amplitude and phase.
•	Preprocessing: Phase sanitization + moving-average filtering + subcarrier selection + normalization.
•	Inference: Fused CSI features → MLP → four-class output.
•	Decision: Fire-related classes trigger the alarm and the result is displayed on the LCD.
#Conclusion
The proposed system provides a lightweight, low-cost, and privacy-preserving solution for NLoS fire and human detection using Wi-Fi CSI. By combining amplitude and sanitized phase with modality-specific subcarrier selection, it reduces feature dimensionality while maintaining high recognition accuracy.
The recommended 10 Hz configuration achieves 96.12% accuracy and offers a practical balance between detection performance and MCU resource consumption. Future work will focus on improving cross-environment robustness and extending real-world deployment.
