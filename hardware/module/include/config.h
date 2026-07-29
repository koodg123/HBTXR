#pragma once

static const int HGTXR_HEIGHT = 256;
static const int HGTXR_WIDTH = 256;
static const int HGTXR_PATCH = 16;
static const int HGTXR_GRID_H = HGTXR_HEIGHT / HGTXR_PATCH;
static const int HGTXR_GRID_W = HGTXR_WIDTH / HGTXR_PATCH;
static const int HGTXR_TOKENS = HGTXR_GRID_H * HGTXR_GRID_W;
static const int HGTXR_EMBED = 192;
static const int HGTXR_STATE = 6;
static const int HGTXR_SEARCH_LOGITS = 7;
static const int HGTXR_TRACK_LOGITS = 8;

