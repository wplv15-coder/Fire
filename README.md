# Overview

The proposed method is a lightweight Non-Line-of-Sight (NLoS) fire and human detection system based on Wi-Fi Channel State Information (CSI), designed for early fire warning in high-rise buildings. It uses CSI amplitude and sanitized phase information to capture wireless channel variations caused by fire and human activity in corridor, through-wall, and cross-floor scenarios.

An ESP32 collects CSI amplitude and phase data and transmits the data to an STM32 through UART. The STM32 performs phase sanitization, moving-average filtering, subcarrier selection, feature normalization, and lightweight MLP classification to distinguish four states: NFN, NFH, FHN, and FWH.

# Highlights

- **Through-Obstacle Detection**: Supports corridor, through-wall, and cross-floor NLoS fire detection.
- **Amplitude-Phase Fusion**: Uses both CSI amplitude and sanitized phase information.
- **MCU-Friendly**: Reduces CSI feature dimensionality for real-time STM32 deployment.
- **Fire + Human Joint Recognition**: Distinguishes NFN, NFH, FHN, and FWH states.
- **Strong Performance**: Achieves **96.12%** detection accuracy at the recommended 10 Hz sampling rate.

# Method

- **Phase Sanitization**: Removes CSI phase distortions caused by hardware and synchronization errors.
- **Moving-Average Filtering**: Uses a filtering-window length of 10 to suppress short-term noise and random fluctuations.
- **SNR-Based Amplitude Selection**: Retains strong and stable amplitude subcarriers.
- **Fisher-Based Phase Selection**: Selects phase subcarriers with stronger class-discrimination capability.
- **Feature Fusion + MLP**: Normalized amplitude and phase features are concatenated and fed into a lightweight MLP with Softmax output for four-class classification.

# Experimental Setup

- **Platform**: ESP32 + STM32, connected via UART.
- **Scenarios**: Corridor, through-wall, and cross-floor NLoS environments.
- **Classes**: NFN, NFH, FHN, and FWH.
- **Best Feature Combination**: 8 amplitude subcarriers + 12 phase subcarriers, reducing the original 104-dimensional CSI representation to 20 dimensions.
- **Ablation Results**: At 100 Hz, amplitude + phase achieves **97.42%** accuracy; without amplitude, **94.28%**; without phase, **93.52%**.
- **Recommended Sampling Rate**: At 10 Hz, the system achieves **96.12%** accuracy while maintaining low RAM, Flash, and inference latency.

# Advantages

- **NLoS Detection Capability** beyond conventional LoS-based sensing.
- **Low Cost** by reusing existing Wi-Fi infrastructure.
- **Privacy-Preserving** sensing without visual information.
- **Human-Aware Detection** reduces confusion between human activity and fire-induced signal changes.
- **Lightweight Deployment** supports real-time inference on STM32.

# System Architecture

- **Acquisition**: ESP32 collects CSI amplitude and phase data.
- **Preprocessing**: Phase sanitization + moving-average filtering + subcarrier selection + normalization.
- **Inference**: Fused CSI features → MLP → four-class output.
- **Decision**: Fire-related classes trigger an alarm, and the detection result is displayed on the LCD.

# Conclusion

The proposed system provides a lightweight, low-cost, and privacy-preserving NLoS fire and human detection solution using Wi-Fi CSI. By combining CSI amplitude with sanitized phase information and applying modality-specific subcarrier selection, the system reduces feature dimensionality while maintaining high recognition accuracy.

Considering both detection performance and MCU resource consumption, a **10 Hz** sampling rate is recommended. Under this configuration, the system achieves **96.12%** detection accuracy, providing a practical balance among accuracy, computation, and memory usage. Future work can further improve robustness across different environments and evaluate larger-scale real-world deployments.
