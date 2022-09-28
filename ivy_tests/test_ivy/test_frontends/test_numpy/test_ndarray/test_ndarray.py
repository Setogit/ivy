# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


# reshape
@st.composite
def dtypes_x_reshape(draw):
    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=10,
            ),
        )
    )
    shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
    return dtypes, x, shape


@handle_cmd_line_args
@given(
    dtypes_x_shape=dtypes_x_reshape(),
)
def test_numpy_ndarray_reshape(
    dtypes_x_shape,
    as_variable,
    native_array,
    fw,
):
    input_dtype, x, shape = dtypes_x_shape
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "shape": shape,
        },
        fw=fw,
        frontend="numpy",
        class_name="ndarray",
        method_name="reshape",
    )


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_numpy_ndarray_add(
    dtype_and_x,
    as_variable,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "value": x[1],
        },
        fw=fw,
        frontend="numpy",
        class_name="ndarray",
        method_name="add",
    )


# transpose
@handle_cmd_line_args
@given(
    array_and_axes=np_frontend_helpers._array_and_axes_permute_helper(
        min_num_dims=2, max_num_dims=5,
        min_dim_size=2, max_dim_size=10,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.transpose"
    ),
)
def test_numpy_ndarray_transpose(
    array_and_axes,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    array, dtype, axes = array_and_axes
    helpers.test_frontend_method(
        input_dtypes_init=dtype,
        input_dtypes_method=dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=num_positional_args,
        num_positional_args_method=num_positional_args,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": np.array(array),
        },
        all_as_kwargs_np_method={
            "axes": axes,
        },
        fw=fw,
        frontend="numpy",
        class_name="ndarray",
        method_name="transpose",
    )