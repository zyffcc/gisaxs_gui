# Independent Fit Window Usage Guide

## Overview
The Independent Fit Window provides enhanced visualization for GISAXS cut analysis results with real-time synchronization and improved matplotlib capabilities.

## Features

### 1. Access Methods
- **Double-click activation**: Double-click anywhere in the `fitGraphicsView` area to open the independent fit window
- **Enhanced visualization**: Larger, clearer display with full matplotlib toolbar
- **Real-time synchronization**: All cut results automatically update in both main UI and independent window

### 2. Window Features
- **Full matplotlib toolbar**: Zoom, pan, save, configure plots
- **Enhanced plot styling**: Larger markers, better colors, professional appearance
- **Statistics display**: Shows data points count, max/min values
- **Responsive updates**: Automatically reflects changes in log scale and normalization settings

### 3. Synchronization
- **Automatic updates**: When you perform a cut operation, both the main `fitGraphicsView` and the independent window update simultaneously
- **Settings sync**: Log scale (X/Y) and normalization options from the main UI are automatically applied to the independent window
- **Real-time consistency**: Changes in the main UI immediately reflect in the independent window if it's open

## Usage Workflow

### Step 1: Perform Cut Analysis
1. Load GISAXS data in the Fitting page
2. Set up cut parameters (center, cutline values)
3. Click "Cut" button to perform the cut operation
4. Results appear in the main `fitGraphicsView`

### Step 2: Open Independent Window
1. Double-click anywhere in the `fitGraphicsView` area
2. The independent fit window opens with the current cut results
3. Use the matplotlib toolbar for detailed analysis

### Step 3: Continue Analysis
1. Modify cut parameters or perform new cuts
2. Both main view and independent window update automatically
3. Toggle log scales or normalization - changes apply to both views instantly

## Technical Details

### Data Handling
- Cut data is stored internally for independent window access
- Supports both horizontal and vertical cuts
- Q-space and pixel-space coordinate systems supported
- Interpolation to 50 data points for smooth visualization

### Window Management
- Independent window can be closed and reopened without data loss
- Window remembers position and size
- Focus management ensures proper keyboard interaction

### Error Handling
- Validates matplotlib availability before opening
- Checks for cut data existence
- Graceful error messages for missing dependencies

## Benefits

1. **Enhanced Analysis**: Larger view for detailed examination of cut results
2. **Professional Tools**: Full matplotlib functionality for advanced analysis
3. **Seamless Workflow**: No need to manually update or sync data
4. **Flexible Viewing**: Keep both views open for comparison and analysis
5. **Export Capabilities**: Use matplotlib toolbar to save high-quality plots

## Requirements
- matplotlib library (automatically checked)
- PyQt5 for window management
- Existing GISAXS cut functionality

## Troubleshooting

### "Missing Library" Warning
- Install matplotlib: `pip install matplotlib`
- Restart the application

### "No Cut Data" Message
- Perform a cut operation first
- Ensure image data is loaded
- Check cut parameters are valid

### Window Not Responding
- Close and reopen by double-clicking `fitGraphicsView` again
- Check system resources and memory usage
