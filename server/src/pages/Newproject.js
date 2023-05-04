import { Link as RouterLink } from 'react-router-dom';
// material
import { Grid, Button, Container, Stack, Typography, Link, Card } from '@mui/material';
import { styled } from '@mui/material/styles';
// components
import Page from '../components/Page';
import Iconify from '../components/Iconify';
import { BlogPostCard, BlogPostsSort, BlogPostsSearch } from '../sections/@dashboard/blog';
import Logo from '../components/Logo';
// mock
import POSTS from '../_mock/blog';
import { NewprojectForm } from '../sections/@dashboard/projects';

// --------------style----------

const RootStyle = styled('div')(({ theme }) => ({
    [theme.breakpoints.up('md')]: {
      display: 'flex',
    },
  }));
  
const HeaderStyle = styled('header')(({ theme }) => ({
top: 0,
zIndex: 9,
lineHeight: 0,
width: '100%',
display: 'flex',
alignItems: 'center',
position: 'absolute',
padding: theme.spacing(3),
justifyContent: 'space-between',
[theme.breakpoints.up('md')]: {
    alignItems: 'flex-start',
    padding: theme.spacing(7, 5, 0, 7),
},
}));

const SectionStyle = styled(Card)(({ theme }) => ({
width: '100%',
maxWidth: 464,
display: 'flex',
flexDirection: 'column',
justifyContent: 'center',
margin: theme.spacing(2, 0, 2, 2),
}));

const ContentStyle = styled('div')(({ theme }) => ({
    maxWidth: 480,
    margin: 'auto',
    minHeight: '100vh',
    display: 'flex',
    justifyContent: 'center',
    flexDirection: 'column',
    padding: theme.spacing(12, 0),
  }));
// ----------------------------------------------------------------------

const SORT_OPTIONS = [
  { value: 'latest', label: 'Latest' },
  { value: 'popular', label: 'Popular' },
  { value: 'oldest', label: 'Oldest' },
];

// ----------------------------------------------------------------------

export default function Newproject() {
  return (
    <Page title="Login">
      <RootStyle>
        <HeaderStyle>
          {/* <Logo />    */}
        <Container maxWidth="sm">
          <ContentStyle>
            <Typography variant="h4" gutterBottom>
              Create a new project
            </Typography>

            <Typography sx={{ color: 'text.secondary', mb: 5 }}>Enter your project details below.</Typography>

            {/* <AuthSocial /> */}

            <NewprojectForm />


          </ContentStyle>
        </Container>
        </HeaderStyle>
      </RootStyle>
    </Page>

  );
}
