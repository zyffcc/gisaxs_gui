# Mouse Selection for Cut Line Parameters

## Overview
The GISAXS GUI now supports interactive mouse selection for setting Cut Line parameters in the fitting controller.

## How to Use

### 1. Open Image in Independent Window
- Import a GISAXS image using the "Import" button
- Click "Show" to display the image 
- **Double-click** on the graphics view to open the independent matplotlib window

### 2. Activate Selection Mode
- In the independent window, **right-click** anywhere to activate selection mode
- The window title will change to indicate selection mode is active
- The cursor will change to a crosshair

### 3. Select Region
- **Left-click and drag** to create a selection rectangle
- The rectangle will be displayed in red as you drag
- Release the mouse button to complete the selection

### 4. Automatic Parameter Update
- The selected region's parameters will automatically be applied to Cut Line controls:
  - Center X/Y coordinates
  - Width and height dimensions
- The selection info will be displayed in the window title and status bar

### 5. Keyboard Shortcuts
- **ESC**: Exit selection mode
- **Delete/Backspace**: Clear current selection
- **Right-click**: Toggle selection mode on/off

## Technical Details

### Coordinate System
- The selection uses pixel coordinates
- Y-coordinates are automatically flipped to match image orientation
- Center coordinates represent the geometric center of the selected rectangle

### UI Control Mapping
The system automatically searches for and updates the following UI controls:
- `cutLineVerticalCenterXValue` / `cutLineCenterXValue`
- `cutLineVerticalCenterYValue` / `cutLineCenterYValue` 
- `cutLineVerticalWidthValue` / `cutLineWidthValue`
- `cutLineVerticalHeightValue` / `cutLineHeightValue`
- `parallelWidthValue` / `parallelHeightValue`
- `parallelCenterXValue` / `parallelCenterYValue`

### Minimum Selection Size
- Selections must be at least 5 pixels in both width and height
- Smaller selections will be ignored

## Benefits
- **Interactive Parameter Setting**: Visually select regions instead of manually entering coordinates
- **Precise Control**: Direct feedback on selection area
- **Non-destructive**: Original image and view state are preserved
- **Integration**: Seamless integration with existing Cut Fitting workflow

## Example Workflow
1. Import GISAXS data file
2. Adjust display settings (log scale, contrast, etc.)
3. Double-click to open independent window
4. Right-click to activate selection mode
5. Draw rectangle around region of interest
6. Parameters are automatically applied to Cut Line controls
7. Continue with fitting analysis using the selected parameters
