import { style } from '@vanilla-extract/css'

import { vars } from '../../../../styles/theme/index.css'

export const imageDisplayMain = style({
  height: '100%',
  width: '100%',
});

export const imageDisplayContainer = style({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
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

export const ImageActionsMain = style({
  position: 'absolute',
  top: vars.spacing.medium,
  left: vars.spacing.medium,
  display: 'flex',
  flexDirection: 'row',
});