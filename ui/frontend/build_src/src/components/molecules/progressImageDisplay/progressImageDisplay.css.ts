import { style } from '@vanilla-extract/css'

export const root = style({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  width: '100%',
  // height: '100%',
  maxHeight: '70%',
  position: 'relative',
  overflow: 'scroll',
  backgroundColor: 'white',
  borderRadius: '4px',
  border: '1px solid #e0e0e0',
  boxSizing: 'border-box',
  transition: 'all 0.3s ease-in-out',
  ':hover': {
    borderColor: '#c0c0c0',
  },
});
