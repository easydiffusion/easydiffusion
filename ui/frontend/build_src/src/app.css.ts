import { style } from '@vanilla-extract/css';

export const AppLayout = style({
  position: 'relative',
  width: '100%',
  height: '100%',
  pointerEvents: 'auto',
  display: 'grid',
  backgroundColor: 'rgb(32, 33, 36)',
  gridTemplateColumns: '360px 1fr',
  gridTemplateRows: '100px 1fr 50px',
  gridTemplateAreas: `
    "header header header"
    "create display display"
    "footer footer footer"
  `,
});

export const HeaderLayout = style({
  gridArea: 'header',
});

export const CreateLayout = style({ 
  gridArea: 'create',
});

export const DisplayLayout = style({
  gridArea: 'display',
});

export const FooterLayout = style({
  gridArea: 'footer',
});