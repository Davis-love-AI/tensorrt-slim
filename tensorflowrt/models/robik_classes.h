/* ============================================================================
# [2017] - Robik AI Ltd - Paul Balanca
# All Rights Reserved.

# NOTICE: All information contained herein is, and remains
# the property of Robik AI Ltd, and its suppliers
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Robik AI Ltd
# and its suppliers and may be covered by U.S., European and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Robik AI Ltd.
# =========================================================================== */
#ifndef TFRT_ROBIK_CLASSES_H
#define TFRT_ROBIK_CLASSES_H

#include <string>
#include <vector>
#include <array>

namespace robik
{
/** List of segmentation descriptions used on the Robik!
 */
inline const std::vector<std::string>& seg_descriptions()
{
    static std::vector<std::string> desc = {
        "Nothing",
        "Car",
        "Van",
        "Truck",
        "Bus",
        "Bicycle",
        "Motorcycle",
        "Person",
        "Road",
        "Sidewalk",
        "Terrain",
        "Traffic Sign",
        "Traffic Light",
        "Vegetation",
        "Building",
        "Sky",
        "Fence",
        "Pole",
        "Parking"
    };
    return desc;
}
/** List of segmentation colors, used for visualization.
 * Returns of vector of RGBA colors.
 */
inline const std::vector<std::array<uint8_t,4>>& seg_colors()
{
    static std::vector<std::array<uint8_t,4>> colors = {
        {0, 0, 0, 0},
        {0, 0, 142, 0},
        {0, 110, 100, 0},
        {30, 30, 70, 0},
        {0, 60, 100, 0},
        {119, 11, 32, 0},
        {50, 0, 230, 0},
        {220, 20, 60, 0},
        {128, 64, 128, 0},
        {244, 35, 232, 0},
        {152, 251, 152, 0},
        {220, 220, 0, 0},
        {250, 170, 30, 0},
        {107, 142, 35, 0},
        {70, 70, 70, 0},
        {70, 130, 180, 0},
        {190, 153, 153, 0},
        {153, 53, 153, 0},
        {250, 170, 160, 0}
    };
    return colors;
}

}

#endif
