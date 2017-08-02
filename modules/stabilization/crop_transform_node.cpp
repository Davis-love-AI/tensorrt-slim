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
#include "vstab_nodes.hpp"

// 0 - x
// 1 - y
// 2 - width
// 3 - height
typedef Eigen::Vector4f Rectanglef;

// 0 - x
// 1 - y
typedef Eigen::Vector2f Point2f;

static const char KERNEL_CROP_STAB_TRANSFORM_NAME[VX_MAX_KERNEL_NAME] = "example.nvx.crop_stab_transform";

static void transformPoint(const Matrix3x3f_rm &H, const Point2f & p, Point2f & newP)
{
    float x = H(0, 0) * p(0) + H(0, 1) * p(1) + H(0, 2);
    float y = H(1, 0) * p(0) + H(1, 1) * p(1) + H(1, 2);
    float z = H(2, 0) * p(0) + H(2, 1) * p(1) + H(2, 2);

    newP(0) = x / z;
    newP(1) = y / z;
}

static bool rectContains(const Rectanglef & rect, const Point2f & pt)
{
    return rect(0) <= pt(0) && pt(0) < rect(0) + rect(2) &&
           rect(1) <= pt(1) && pt(1) < rect(1) + rect(3);
}

static bool isPointInsideCroppingRect(const Rectanglef & rect, const Matrix3x3f_rm & H, const Point2f & p)
{
    Point2f newP;
    transformPoint(H, p, newP);

    return rectContains(rect, newP);
}

static bool isMotionGood(const Matrix3x3f_rm & transform,
                         int frameWidth, int frameHeight,
                         const Matrix3x3f_rm & resizeMat, float factor)
{
    Rectanglef rect;
    rect << 0.0f, 0.0f,
            static_cast<float>(frameWidth - 1),
            static_cast<float>(frameHeight - 1);

    Matrix3x3f_rm H = (1 - factor) * transform + factor * resizeMat;

    Point2f p1, p2, p3, p4;
    p1 << 0.0f, 0.0f;
    p2 << static_cast<float>(frameWidth - 1), 0.0f;
    p3 << static_cast<float>(frameWidth - 1), static_cast<float>(frameHeight - 1);
    p4 << 0.0f, static_cast<float>(frameHeight - 1);

    return isPointInsideCroppingRect(rect, H, p1) && isPointInsideCroppingRect(rect, H, p2) &&
           isPointInsideCroppingRect(rect, H, p3) && isPointInsideCroppingRect(rect, H, p4);
}

static bool cropTransform(const Matrix3x3f_rm & transform, int frameWidth, int frameHeight,
                              const Matrix3x3f_rm & resizeMat, Matrix3x3f_rm & truncatedTransform)
{
    float t = 0;
    if ( isMotionGood(transform, frameWidth, frameHeight, resizeMat, t) )
    {
        return false;
    }

    float l = 0, r = 1;
    while (r - l > 1e-2f)
    {
        t = (l + r) * 0.5f;
        if ( isMotionGood(transform, frameWidth, frameHeight, resizeMat, t) )
            r = t;
        else
            l = t;
    }

    truncatedTransform = (1 - t) * transform + t * resizeMat;

    return true;
}

// Kernel implementation
static vx_status VX_CALLBACK cropStabTransform_kernel(vx_node, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 4)
        return VX_FAILURE;

    vx_status status = VX_SUCCESS;

    vx_matrix vxStabTransform = (vx_matrix)parameters[0];
    vx_matrix vxTruncatedTransform = (vx_matrix)parameters[1];
    vx_image image = (vx_image)parameters[2];
    vx_scalar sCropMargin = (vx_scalar)parameters[3];

    vx_float32 stabTransformData[9] = {0};
    status |= vxCopyMatrix(vxStabTransform, stabTransformData, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    Matrix3x3f_rm stabTransform = Matrix3x3f_rm::Map(stabTransformData, 3, 3), invStabTransform;

    vx_float32 cropMargin;
    status |= vxCopyScalar(sCropMargin, &cropMargin, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if (cropMargin < 0) // without truncation
    {
        invStabTransform = stabTransform.inverse(); // inverse the matrix for vxWarpPerspectiveNode
        status |= vxCopyMatrix(vxTruncatedTransform, invStabTransform.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        return status;
    }

    vx_uint32 width = 0, height = 0;
    status |= vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width));
    status |= vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height));

    Matrix3x3f_rm resizeMat = Matrix3x3f_rm::Identity();
    float scale = 1.0f / (1.0f - 2 * cropMargin);
    resizeMat(0, 0) = resizeMat(1, 1) = scale;
    resizeMat(0, 2) = - scale * width * cropMargin;
    resizeMat(1, 2) = - scale * height * cropMargin;

    stabTransform.transposeInPlace(); // transpose to the standart form like resizeMat
    stabTransform = resizeMat * stabTransform;

    invStabTransform = stabTransform.inverse();
    Matrix3x3f_rm invResizeMat = resizeMat.inverse();

    Matrix3x3f_rm invTruncatedTransform;
    bool isTruncated = cropTransform(invStabTransform, width, height, invResizeMat, invTruncatedTransform);

    if (isTruncated)
    {
        stabTransform = invTruncatedTransform.inverse();
    }

    stabTransform.transposeInPlace(); // inverse transpose
    invStabTransform = stabTransform.inverse(); // inverse the matrix for vxWarpPerspectiveNode
    status |= vxCopyMatrix(vxTruncatedTransform, invStabTransform.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    return status;
}

/** Validate the parameters of node.
 */
static vx_status VX_CALLBACK cropStabTransform_validate(
    vx_node, const vx_reference parameters[], vx_uint32 numParams, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    // Number: stab + crop tr, image, crop box.
    if(numParams != 7)  return VX_ERROR_INVALID_PARAMETERS;

    vx_matrix stab_transform = (vx_matrix)parameters[0];
    // ''ymin, xmin, ymax, xmax'' format for margin scalars.
    vx_scalar crop_top = (vx_scalar)parameters[3];
    vx_scalar crop_left = (vx_scalar)parameters[4];
    vx_scalar crop_bottom = (vx_scalar)parameters[5];
    vx_scalar crop_right = (vx_scalar)parameters[6];

    // Check stabilization matrix format.
    vx_enum stabTransformDataType = 0;
    vx_size stabTransformRows = 0ul, stabTransformCols = 0ul;
    vxQueryMatrix(stab_transform, VX_MATRIX_ATTRIBUTE_TYPE, &stabTransformDataType, sizeof(stabTransformDataType));
    vxQueryMatrix(stab_transform, VX_MATRIX_ATTRIBUTE_ROWS, &stabTransformRows, sizeof(stabTransformRows));
    vxQueryMatrix(stab_transform, VX_MATRIX_ATTRIBUTE_COLUMNS, &stabTransformCols, sizeof(stabTransformCols));
    if (stabTransformDataType != VX_TYPE_FLOAT32 ||
        stabTransformCols != 3 || stabTransformRows != 3) {
        status = VX_ERROR_INVALID_PARAMETERS;
    }

    // Check cropping parameters.
    vx_enum crop_type = 0;
    vxQueryScalar(crop_top, VX_SCALAR_ATTRIBUTE_TYPE, &crop_type, sizeof(crop_type));
    if (crop_type != VX_TYPE_FLOAT32)   status = VX_ERROR_INVALID_TYPE;
    vxQueryScalar(crop_left, VX_SCALAR_ATTRIBUTE_TYPE, &crop_type, sizeof(crop_type));
    if (crop_type != VX_TYPE_FLOAT32)   status = VX_ERROR_INVALID_TYPE;
    vxQueryScalar(crop_bottom, VX_SCALAR_ATTRIBUTE_TYPE, &crop_type, sizeof(crop_type));
    if (crop_type != VX_TYPE_FLOAT32)   status = VX_ERROR_INVALID_TYPE;
    vxQueryScalar(crop_right, VX_SCALAR_ATTRIBUTE_TYPE, &crop_type, sizeof(crop_type));
    if (crop_type != VX_TYPE_FLOAT32)   status = VX_ERROR_INVALID_TYPE;


    vx_meta_format cropTransformMeta = metas[1];
    vx_enum cropTransformType = VX_TYPE_FLOAT32;
    vx_size cropTransformRows = 3;
    vx_size cropTransformCols = 3;

    vxSetMetaFormatAttribute(cropTransformMeta, VX_MATRIX_ATTRIBUTE_TYPE, &cropTransformType, sizeof(cropTransformType));
    vxSetMetaFormatAttribute(cropTransformMeta, VX_MATRIX_ATTRIBUTE_ROWS, &cropTransformRows, sizeof(cropTransformRows));
    vxSetMetaFormatAttribute(cropTransformMeta, VX_MATRIX_ATTRIBUTE_COLUMNS, &cropTransformCols, sizeof(cropTransformCols));
    return status;
}

// Register user defined kernel in OpenVX context
vx_status registerCropStabTransformKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_enum id;
    status = vxAllocateUserKernelId(context, &id);
    if (status != VX_SUCCESS) {
        vxAddLogEntry((vx_reference)context, status,
            "[%s:%u] Failed to allocate an ID for the CropStabTransform kernel",
            __FUNCTION__, __LINE__);
        return status;
    }
    vx_kernel kernel = vxAddUserKernel(context, KERNEL_CROP_STAB_TRANSFORM_NAME,
                                       id,
                                       cropStabTransform_kernel,
                                       4,
                                       cropStabTransform_validate,
                                       NULL,
                                       NULL
                                       );
    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS) {
        vxAddLogEntry((vx_reference)context, status,
            "[%s:%u] Failed to create CropStabTransform Kernel",
            __FUNCTION__, __LINE__);
        return status;
    }

    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED);  // stabTransform
    status |= vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // truncatedTransform
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);   // image
    status |= vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);  // crop top
    status |= vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);  // crop left
    status |= vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);  // crop bottom
    status |= vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);  // crop right
    if (status != VX_SUCCESS) {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status,
            "[%s:%u] Failed to initialize CropStabTransform Kernel parameters",
            __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    status = vxFinalizeKernel(kernel);
    vxReleaseKernel(&kernel);
    if (status != VX_SUCCESS) {
        vxAddLogEntry((vx_reference)context, status,
            "[%s:%u] Failed to finalize CropStabTransform Kernel",
            __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }
    return status;
}

vx_node cropStabTransformNode(
    vx_graph graph,
    vx_matrix stab_transform, vx_matrix crop_transform, vx_image image,
    vx_scalar crop_top, vx_scalar crop_left, vx_scalar crop_bottom, vx_scalar crop_right)
{
    vx_node node = NULL;
    vx_kernel kernel = vxGetKernelByName(
        vxGetContext((vx_reference)graph), KERNEL_CROP_STAB_TRANSFORM_NAME);

    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS) {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);
        if (vxGetStatus((vx_reference)node) == VX_SUCCESS) {
            vxSetParameterByIndex(node, 0, (vx_reference)stab_transform);
            vxSetParameterByIndex(node, 1, (vx_reference)crop_transform);
            vxSetParameterByIndex(node, 2, (vx_reference)image);
            vxSetParameterByIndex(node, 3, (vx_reference)crop_top);
            vxSetParameterByIndex(node, 4, (vx_reference)crop_left);
            vxSetParameterByIndex(node, 5, (vx_reference)crop_bottom);
            vxSetParameterByIndex(node, 6, (vx_reference)crop_right);
        }
    }

    return node;
}
