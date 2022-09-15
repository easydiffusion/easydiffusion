
import { globalStyle } from '@vanilla-extract/css';

globalStyle('body', {
  margin: 0,
  minWidth: '320px',
  minHeight: '100vh',
});

globalStyle('#root', {
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100vw',
  height: '100vh',
});

