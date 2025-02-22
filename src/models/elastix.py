import itk


def register_label(fixed_image, moving_image, moving_label, elx_params):
    """ Register a pair of images and transform a label map. """
    elx = itk.ElastixRegistrationMethod.New(itk.image_from_array(fixed_image), itk.image_from_array(moving_image))
    elx.SetParameterObject(elx_params)
    elx.LogToConsoleOff()

    # Run registration
    elx.Update()

    # Get transformed image
    #result_image = elx.GetOutput()

    # Set transform parameters for labels
    tfx_params = elx.GetTransformParameterObject()
    tfx_params.SetParameter(1, 'FinalBSplineInterpolationOrder', '0')  # Change the last parameter map
    tfx = itk.TransformixFilter.New(itk.image_from_array(moving_label))
    tfx.SetTransformParameterObject(tfx_params)
    tfx.SetLogToConsole(False)

    # Transform moving label
    tfx.Update()

    # Get transformed label as numpy array
    result_label = itk.GetArrayFromImage(tfx.GetOutput())

    return result_label
