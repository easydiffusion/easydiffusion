import { style, globalStyle } from '@vanilla-extract/css'

import { vars } from '../../../../styles/theme/index.css'

export const imageDisplayMain = style({
  height: '100%',
  width: '100%',
  display: 'flex',
  flexDirection: 'column',
});

export const imageDisplayContainer = style({
  height: '100%',
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
  position: 'relative',
});

globalStyle(`${imageDisplayContent} > div`, {
  marginBottom: vars.spacing.medium,
});

globalStyle(`${imageDisplayContent} p`, {
  marginBottom: vars.spacing.small,
});

globalStyle(`${imageDisplayContent} button`, {
  marginRight: vars.spacing.medium,
});

export const ImageActions = style({
  position: 'absolute',
  top: '0',
  left: '0',
  display: 'flex',
  flexDirection: 'row',
});