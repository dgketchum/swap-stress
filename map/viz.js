/**
 * Visualization Script for SWAP-STRESS Project Data
 * 
 * Description: This script loads and visualizes the key data layers used in the
 *              stack_bands_climatology function. It is intended for use in the
 *              Google Earth Engine Code Editor for data exploration.
 * Author: Gemini
 * Date: 19-08-2025
 */

// --- ROI and Map Setup ---
// Define a central point of interest for the map.
var roi = ee.Geometry.Point([-105.5, 40.5]);
Map.centerObject(roi, 7);

// --- Base Layer (Recent Landsat Imagery) ---
var landsat_image = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(Map.getBounds(true))
    .filterDate('2022-05-01', '2022-09-30')
    .map(function(image) {
        // Applies scaling factors.
        var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
        return image.addBands(opticalBands, null, true);
    })
    .median();

var landsat_vis = {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],
  min: 0.0,
  max: 0.2,
};
Map.addLayer(landsat_image, landsat_vis, 'Landsat (2022)');

// --- Dynamic Climatology Layers (Examples) ---

// Sentinel-1 Radar
var s1_mean = ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterDate('2022-01-01', '2022-12-31')
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .select('VV')
               .mean();
Map.addLayer(s1_mean, {min: -15, max: 0}, 'Sentinel-1 VV Mean (2022)', false);

// PRISM Climate Data
var prism = ee.Image("OREGONSTATE/PRISM/Norm91m").select('ppt');
var prism_vis = {
  min: 200, max: 1500, 
  palette: ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']
};
Map.addLayer(prism, prism_vis, 'PRISM PPT', false);


// --- Terrain Layers ---
var ned = ee.Image('USGS/3DEP/10m');
var elevation = ned.select('elevation');
var slope = ee.Terrain.slope(elevation);
var twi = ee.Image("users/zhoylman/CONUS_TWI_epsg5072_30m");

Map.addLayer(elevation, {min: 1000, max: 3500, palette: ['#f7fcfd','#e0ecf4','#bfd3e6','#9ebcda','#8c96c6','#8856a7','#810f7c']}, 'Elevation', false);
Map.addLayer(slope, {min: 0, max: 45}, 'Slope', false);
Map.addLayer(twi, {min: 5, max: 25, palette: ['#d73027','#fc8d59','#fee090','#e0f3f8','#91bfdb','#4575b4']}, 'Topographic Wetness Index', false);

// --- Soil Layers ---

// SSURGO
var ssurgo_sand = ee.Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite');
Map.addLayer(ssurgo_sand, {min: 10, max: 70, palette: ['#d53e4f','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5']}, 'SSURGO Sand', false);

// POLARIS
var polaris_clay = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').mean();
Map.addLayer(polaris_clay, {min: 10, max: 40, palette: ['#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8']}, 'POLARIS Clay', false);

// --- Land Cover & Land Use Layers ---

// NLCD
var nlcd = ee.Image('USGS/NLCD/NLCD2019').select('landcover');
// Visualization parameters are built-in for NLCD
Map.addLayer(nlcd, {}, 'NLCD', false);

// Global Surface Water
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
Map.addLayer(gsw, {min: 0, max: 100, palette: ['#ffffff', '#0000ff']}, 'Global Surface Water', false);

// Landform
var landform = ee.Image('projects/usgs-gap/landform').select('landform');
var landform_vis = {
  min: 11, max: 42, 
  palette: [
    '#141414', '#383838', '#808080', '#EBEB8F', '#F7D311', '#F7A278', '#C9643B',
    '#C93214', '#969696', '#646464', '#C8C8C8', '#F7F2E0', '#F7E0C8', '#EBEB8F',
    '#d7c259', '#a08214', '#785a00', '#463200', '#e0e0e0', '#d2d2d2', '#a0a0a0',
    '#828282', '#646464', '#464646', '#282828', '#000000'
  ]
};
Map.addLayer(landform, landform_vis, 'Landform', false);
