import { style } from '@vanilla-extract/css'

export const InpaintingPanelMain = style({
  position: 'relative',
  width: '100%',
  height: '100%',
  padding: '10px 10px',
});

export const InpaintingControls = style({
  display: 'flex',
  flexDirection: 'row',
  width: '100%',
  flexWrap: 'wrap',
});

export const InpaintingControlRow = style({
  display: 'flex',
  flexDirection: 'row',
  justifyContent: 'space-evenly',
  alignItems: 'center',
  width: '100%',

  ':first-of-type': {
    margin: '10px 0',

  },

});
