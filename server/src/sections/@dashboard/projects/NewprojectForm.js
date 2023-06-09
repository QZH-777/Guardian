import * as Yup from 'yup';
import { useState } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { useFormik, Form, FormikProvider } from 'formik';
// material
import { Link, Stack, Checkbox, TextField, IconButton, InputAdornment, FormControlLabel ,Fab} from '@mui/material';
import { LoadingButton } from '@mui/lab';
// // component
// import Iconify from '../../../components/Iconify';
// icon
import NavigationIcon from '@mui/icons-material/Navigation';

// ----------------------------------------------------------------------

export default function LoginForm() {
  const navigate = useNavigate();

  const [showPassword, setShowPassword] = useState(false);

  const LoginSchema = Yup.object().shape({
    email: Yup.string().email('Email must be a valid email address').required('Email is required'),
    password: Yup.string().required('Password is required'),
  });

  const formik = useFormik({
    initialValues: {
      email: '',
      password: '',
      remember: true,
    },
    validationSchema: LoginSchema,
    onSubmit: () => {
      navigate('/dashboard', { replace: true });
    },
  });

  const { errors, touched, values, isSubmitting, handleSubmit, getFieldProps } = formik;

  const handleShowPassword = () => {
    setShowPassword((show) => !show);
  };

  return (
    <FormikProvider value={formik}>
      <Form autoComplete="off" noValidate onSubmit={handleSubmit} spacing={3}>
        <Stack spacing={3}>
          {/* <TextField
            fullWidth
            autoComplete="username"
            type="email"
            label="Email address"
            {...getFieldProps('email')}
            error={Boolean(touched.email && errors.email)}
            helperText={touched.email && errors.email}
          /> */}
          <TextField
            fullWidth
            autoComplete="projectname"
            type="email"
            label="Project Name"
            // {...getFieldProps('email')}
            // error={Boolean(touched.email && errors.email)}
            // helperText={touched.email && errors.email}
          />

          {/* <TextField
            fullWidth
            autoComplete="current-password"
            type={showPassword ? 'text' : 'password'}
            label="Password"
            {...getFieldProps('password')}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton onClick={handleShowPassword} edge="end">
                    <Iconify icon={showPassword ? 'eva:eye-fill' : 'eva:eye-off-fill'} />
                  </IconButton>
                </InputAdornment>
              ),
            }}
            error={Boolean(touched.password && errors.password)}
            helperText={touched.password && errors.password}
          /> */}

            
            <Fab variant="extended">
                <NavigationIcon sx={{ mr: 1 }} />
                Upload Data
            </Fab> 
        <TextField
            fullWidth
            autoComplete="projectname"
            type="email"
            label="Project Describe"
            multiline
            rows={4}
            // {...getFieldProps('email')}
            // error={Boolean(touched.email && errors.email)}
            // helperText={touched.email && errors.email}
          />
            
        </Stack>

        {/* <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ my: 2 }}>
          <FormControlLabel
            control={<Checkbox {...getFieldProps('remember')} checked={values.remember} />}
            label=""
          />

        </Stack> */}

        <LoadingButton fullWidth size="large" type="submit" variant="contained" loading={isSubmitting} sx={{ p: 2 }}  >
          Create
        </LoadingButton>
      </Form>
    </FormikProvider>
  );
}
