import { style, globalStyle } from '@vanilla-extract/css'


import { vars } from '../../../../../styles/theme/index.css'

export const imageDisplayMain = style({
  height: '100%',
  width: '100%',
  display: 'flex',
  flexDirection: 'column',
});

export const imageDisplayContainer = style({
  height: '100%',
  width: '80%',
  display: 'flex',
  justifyContent: 'center',
});

export const imageDisplayCenter = style({
  width: '100%',
  maxWidth: '1000px',
  position: 'relative',
});

export const imageDisplayContent = style({
  display: 'flex',
  flexDirection: 'column',
});

globalStyle(`${imageDisplayContent} > div`, {
  marginBottom: vars.spacing.large,
});

globalStyle(`${imageDisplayContent} p`, {
  marginBottom: vars.spacing.small,
});

globalStyle(`${imageDisplayContent} button`, {
  marginRight: vars.spacing.medium,
});
