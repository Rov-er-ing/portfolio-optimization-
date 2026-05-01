# Design System: ML Portfolio Optimization (MLPO)
**Project ID:** 4044680601076201931

## 1. Visual Theme & Atmosphere
The **"Institutional Dark"** aesthetic is designed for high-performance financial environments. The atmosphere is **dense, high-contrast, and utilitarian**, prioritizing data clarity and immediate pattern recognition. It avoids decorative elements, favoring a "command-center" philosophy where every pixel serves a functional purpose.

## 2. Color Palette & Roles
*   **Obsidian Deep Black (#0A0A0A)**: The foundational canvas. Provides the absolute contrast required for luminous data overlays.
*   **Neon Optimization Green (#00FF9D)**: The primary accent and signal for "Optimized" states. Used for the primary "Optimize" button, positive returns, and the Efficient Frontier highlights.
*   **Muted Slate (#94A3B8)**: The functional neutral. Used for labels, secondary metadata, and axis lines in charts to prevent visual fatigue.
*   **Vibrant Alert Red (#EF4444)**: The high-priority signal. Reserved strictly for negative returns, risk breach alerts, and system errors.
*   **Steel Stroke (#1E293B)**: Used for container borders and subtle grid dividers.

## 3. Typography Rules
*   **Font Family**: **Inter** (Sans-serif) for primary UI elements and **Geist Mono** for mathematical outputs, ticker symbols, and weights.
*   **Weights**: Light (300) for labels to maintain elegance; Semibold (600) for headers and primary values to ensure legibility in high-density views.
*   **Letter Spacing**: Subtly tracked-out (0.02em) for mono values to enhance readability of financial digits.

## 4. Component Stylings
*   **Buttons**: Sharp 4px corners with a subtle inner-glow effect when active. Secondary buttons use a transparent ghost-style with Steel Stroke borders.
*   **Cards/Containers**: Background Obsidian (#0A0A0A) with a precise 1px Steel Stroke (#1E293B) border. No shadows; elevation is communicated through border intensity.
*   **Inputs/Forms**: Deep Charcoal (#171717) backgrounds with Muted Slate labels. Focus state is signaled by a Neon Green (#00FF9D) border.

## 5. Layout Principles
*   **Density**: High-density grid-based layout (Bento Box style). Minimal margins (12px) to maximize screen real-estate for chart visualization.
*   **Alignment**: Strict baseline alignment for numerical data. All financial figures are right-aligned or decimal-aligned for rapid comparative scanning.
