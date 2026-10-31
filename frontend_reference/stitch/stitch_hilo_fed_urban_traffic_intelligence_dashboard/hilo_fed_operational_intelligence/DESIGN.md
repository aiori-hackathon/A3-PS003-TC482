---
name: HiLo-Fed Operational Intelligence
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#434655'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#737686'
  outline-variant: '#c3c6d7'
  surface-tint: '#0053db'
  primary: '#004ac6'
  on-primary: '#ffffff'
  primary-container: '#2563eb'
  on-primary-container: '#eeefff'
  inverse-primary: '#b4c5ff'
  secondary: '#006a61'
  on-secondary: '#ffffff'
  secondary-container: '#86f2e4'
  on-secondary-container: '#006f66'
  tertiary: '#006056'
  on-tertiary: '#ffffff'
  tertiary-container: '#007b6e'
  on-tertiary-container: '#b1fff1'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dbe1ff'
  primary-fixed-dim: '#b4c5ff'
  on-primary-fixed: '#00174b'
  on-primary-fixed-variant: '#003ea8'
  secondary-fixed: '#89f5e7'
  secondary-fixed-dim: '#6bd8cb'
  on-secondary-fixed: '#00201d'
  on-secondary-fixed-variant: '#005049'
  tertiary-fixed: '#71f8e4'
  tertiary-fixed-dim: '#4fdbc8'
  on-tertiary-fixed: '#00201c'
  on-tertiary-fixed-variant: '#005048'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.025em
  headline-xl-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 26px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 22px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.04em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.05em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.06em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-sm: 0.5rem
  gutter-lg: 1.5rem
  margin: 1.5rem
  margin-sm: 1rem
  margin-lg: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style
The design system combines the rigorous technical authority of mission-critical aerospace command centers with the clarity of a high-end enterprise research platform. It delivers an operational cockpit optimized for deep spatial awareness, multi-sensor telemetry, and real-time civic intelligence analysis. 

The aesthetic is precision-engineered, clinical, and high-fidelity. By merging a light, luminous tactical canvas with ultra-fine hairpins, crisp tabular figures, and calibrated chromatic telemetry indicators, the interface achieves the weightless, high-bandwidth density of Palantir-style software without cognitive clutter. The platform commands confidence, technical speed, and analytical precision.

## Colors
The system employs a luminous tactical light palette engineered for all-day operational control, complemented by an inverse high-contrast dark mode for surveillance centers and low-light field deployments.

- **Primary Canvas & Structures:** Default canvas rests on soft off-white (`#F8FAFC`). Elevated analytical panels and control cards sit on pure white (`#FFFFFF`). Hairline boundaries are enforced with subtle steel borders (`#E2E8F0` light, `#334155` dark).
- **Core Brand Accents:** Electric Cobalt (`#2563EB`) directs cognitive focus toward real-time action hooks, primary triggers, and active vector pathways. Tactical Teal (`#0D9488` / `#14B8A6`) highlights sensor feeds, telemetry tracks, and geospatial metadata.
- **Typography & Neutrals:** Deepest Navy (`#0F172A`) commands high-contrast headline reading, complemented by Slate Navy (`#1E293B`) for primary operational data, and Muted Slate (`#64748B`) for metadata labels and inactive boundaries.
- **Operational Status (Telemetry Tiers):**
  - **Nominal / Optimal:** Emerald (`#10B981`)
  - **Cautionary / Sensor Drift:** Amber (`#F59E0B`)
  - **Critical Alert / Breach:** Crimson (`#EF4444`)

## Typography
Typography is split purposefully between structured interface text and tactical machine telemetry:
- **Inter** provides high legibility, neutral geometric forms, and high structural density across interfaces, dashboards, and modal workflows. Tight negative letter spacing is deployed on display headers to mimic mission flight-deck readouts.
- **JetBrains Mono** is assigned to labels, sensor IDs, latitude/longitude readouts, system state badges, and real-time feeds. Monospaced rendering ensures aligned tabular data streams with zero horizontal layout shift during numeric fluctuations.
- All body copies enforce `tabular-nums` OpenType font features to ensure precision data alignment across live-updating sensor streams.

## Layout & Spacing
The layout adheres to a high-density, 12-column fluid grid tailored for tactical command displays, split-screen geospatial maps, and analytics drawers.

- **Grid Architecture:** 
  - **Desktop (>= 1280px):** 12-column fluid grid, 16px gutters, 24px canvas margins. Side-panel toolbars pin to 280px / 360px static dockings while main data viewports adapt flexibly.
  - **Tablet (768px - 1279px):** 8-column layout, 12px gutters, 16px canvas margins. Secondary sidebars collapse into overlay sliding draw sheets.
  - **Mobile (< 768px):** 4-column layout, 8px gutters, 12px margins. Tactical tabs convert into segmented toggle ribbons.
- **Spacing Rhythm:** Rhythms prioritize compact efficiency (`0.25rem` to `1.5rem`). Information density remains tight to maximize real-time metric exposure per visual frame while maintaining absolute hairline breathing room.

## Elevation & Depth
Elevation in the platform is achieved through architectural layering, delicate ambient multidirectional drop shadows, and ultra-crisp hairline delimiters rather than heavy drop shadows.

- **Baseline Level (Canvas):** Flat `#F8FAFC`, non-elevated ground plane for base map viewports and foundational canvas tiles.
- **Level 1 (Docked Containers & Tables):** Pure `#FFFFFF` background, bounded by a uniform `1px solid #E2E8F0` border. Shadow: `0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02)`.
- **Level 2 (Floating Floating Inspect Panels & Toolbars):** Pure `#FFFFFF` background with 95% opacity blur overlay (`backdrop-filter: blur(8px)`). Shadow: `0 4px 6px -1px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04)`.
- **Level 3 (Tactical Modals & Command Palettes):** Multi-tiered projection shadow with slate tint: `0 20px 25px -5px rgba(15, 23, 42, 0.08), 0 8px 10px -6px rgba(15, 23, 42, 0.04)`, encircled by a reinforced structural border `1px solid #CBD5E1`.

## Shapes
To preserve an engineered, instrumentation-grade feel, corners are kept deliberately tight and disciplined.

- Standard buttons, badges, status chips, and form fields adopt a refined `rounded` (4px) profile.
- Major data panels, telemetry cards, and flyout sheets use `rounded-lg` (8px).
- Modals, tactical command bars, and main floating map control pods cap at `rounded-xl` (12px).
- Fully circular shapes (`rounded-full`) are reserved exclusively for radar beacon pulses, ping indicators, and avatar identifiers.

## Components

### Buttons
- **Primary Operational:** High-voltage `#2563EB` fill, white `#FFFFFF` text, `font-medium`, 1px solid `#1D4ED8` border. Slight hover brightening (`#3B82F6`). Active state drops into a crisp inset micro-shadow.
- **Secondary / Ghost:** Transparent canvas or `#FFFFFF`, `1px solid #E2E8F0`, deep slate `#1E293B` text. On hover, background shifts to `#F1F5F9` with sharp `#CBD5E1` outline.
- **Telemetry Action (Teal):** `#0D9488` background, `#FFFFFF` text, used for geo-spatial spatial querying, sensor syncs, and telemetry export triggers.
- **Dimensions:** Compact 32px height standard for desktop dashboards; 40px for mobile/tablet interactive touch targets.

### Badges & Status Chips
- Built with JetBrains Mono, uppercase `label-sm` sizing, and a fixed height of 20px with 6px horizontal padding.
- Employs light tonal surfaces with saturated semantic borders and dot indicators:
  - **Nominal:** Background `#ECFDF5`, border `#A7F3D0`, text `#065F46`, live pulse dot `#10B981`.
  - **Warning:** Background `#FFFBEB`, border `#FDE68A`, text `#92400E`, live pulse dot `#F59E0B`.
  - **Alert:** Background `#FEF2F2`, border `#FECACA`, text `#991B1B`, live pulse dot `#EF4444`.

### Tables & Dense Lists
- Striped or bordered layout using ultra-fine `1px solid #F1F5F9` horizontal row separators.
- Header row uses uppercase `label-md` JetBrains Mono text in `#64748B`, set over an off-white sub-tier background `#F8FAFC`.
- Row height is constrained to 36px for high density, expanding to 44px on relaxed modes. Hover state shifts instantly to `#F8FAFC`.

### Form Fields & Inputs
- Height 32px or 36px. Pure white background, `1px solid #CBD5E1` frame, `rounded` 4px geometry.
- Focus state activates an electric focus ring: `outline: none; border-color: #2563EB; box-shadow: 0 0 0 1px #2563EB`.
- Value font defaults to Inter 13px, with unit suffixes (e.g., `ms`, `km/h`, `lux`) styled in JetBrains Mono 11px muted `#64748B`.

### Checkboxes & Segmented Controls
- Checkboxes: 14x14px square with 2px corner radius, `1px solid #94A3B8`. Active check switches to electric blue `#2563EB` with an ultra-fine white icon.
- Segmented Control Switcher: Encased in a `#F1F5F9` tray; active segment floats on top with crisp white `#FFFFFF`, `1px solid #CBD5E1`, and a 1px ambient drop shadow.

### Cards & Telemetry Panels
- Floating white card container `#FFFFFF` with `1px solid #E2E8F0` framing.
- Headers are demarcated with an explicit internal 36px header bar containing title, live sensor status dot, and icon controls, separated by an inner border rule.
- Embedded telemetry sparklines and metrics incorporate Tactical Teal (`#0D9488`) fills and Electric Blue trend lines.