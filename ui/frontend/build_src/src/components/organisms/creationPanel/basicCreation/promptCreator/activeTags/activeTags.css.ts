import { style } from '@vanilla-extract/css';
import { vars } from '../../../../../../styles/theme/index.css';
export const ActiveTagListMain = style({
  display: 'flex',
  flexDirection: 'row',
  flexWrap: 'wrap',
  gap: '10px',
  width: '100%',
  height: '100%',
  overflow: 'visible',
  scrollbarWidth: 'none',
  msOverflowStyle: 'none',
  '::-webkit-scrollbar': {
    display: 'none',
  },
});


