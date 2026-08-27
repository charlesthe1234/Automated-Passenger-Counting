# CAG Dashboard Style Guide

## Direction

The V1 dashboard supports dark and light operational monitoring styles:

- dark mode uses a dark slate background
- light mode uses a white/light grey page background with dark readable text
- high-contrast live video panel
- compact right sidebar
- compact camera switcher inside the video panel when multiple cameras exist
- schematic tactical floor map for live X/Y person dots from the CV homography pipeline
- boundary-aware tactical map: red dots mean inside occupancy, cyan dots mean outside visible context
- separate Passenger Assistance tab for filtered identity cards and five-view evidence galleries
- color only for status, alert severity, and connection state
- no landing-page or marketing layout
- theme controls must visibly contrast in both modes; light-mode icons need dark outlines/text

## Layout

Primary hierarchy:

1. Header with dashboard name and overall camera connection state.
2. Tab navigation for Operations and Passenger Assistance.
3. Operations tab: compact status pill row, tactical floor map, zone capacity bars, 60-minute trend sparkline, large live MJPEG video panel, and Metrics/Alerts sidebar mini-tabs.
4. Assistance tab: demographics totals first, filters second, then compact one-person identity cards. The primary thumbnail should prefer Front, then a side view, Baseline, and Back.

## Component Rules

- `VideoPlayer` owns the selected camera stream display and the camera selector.
- `OperationsStatusPills` shows backend, selected camera health, all-camera count, and resolution above the video.
- `TacticalMap` shows the fused X/Y positions on a schematic map, keeps the calibrated tent as the dominant central region, compresses outside visible context into a thin border, and marks stale data clearly.
- `TacticalMap` legends must stay small: red = inside occupancy, cyan = outside visible. Outside visible does not affect capacity.
- `ZoneCapacityBars` shows camera-keyed capacity pressure between the status pills and the live video.
- `MetricTrendSparkline` shows a compact trailing passenger-count trend beside the zone capacity panel.
- `OperationsSidebarTabs` toggles between metrics and alerts in the right sidebar.
- `MetricsPanel` stays compact with latest 10 rows.
- `AlertsPanel` stays compact with latest 5 alerts and severity indicators.
- `AssistanceView` displays backend-powered demographic totals and one card per ReID master identity as an assistance filter, not identification.
- Identity thumbnails open a focused modal with Front, Left, Right, Back, and Baseline slots. Missing views remain visibly marked instead of duplicating another angle.
- The gallery modal uses a large inspectable image, a compact five-thumbnail strip, capture context, and a persistent manual-verification notice.
- Offline video state must follow the selected camera status from `/api/cameras`, not image error events.

## Update Rule

Update this file whenever dashboard visual direction, first-screen hierarchy, or major UI component responsibilities change.
