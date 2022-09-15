import { style } from '@vanilla-extract/css';

export const ImageInputDisplay = style({
  display: 'flex',
  // justifyContent:'space-around',
});

export const InputLabel = style({
  marginBottom: '5px',
  display:'block'

});

export const ImageInput = style({
  display:'none',
});

export const ImageInputButton = style({
  backgroundColor: 'rgb(38, 77, 141)',
  fontSize: '1.2em',
  fontWeight: 'bold',
  color: 'white',
  padding:'8px',
  borderRadius: '5px',
});

// this is needed to fix an issue with the image input text
// when that is a drag an drop we can remove this
export const ImageFixer = style({
  marginLeft: '20px',
});

export const XButton = style({
  position: 'absolute',
  transform: 'translateX(-50%) translateY(-35%)',
  background: 'black',
  color: 'white',
  border: '2pt solid #ccc',
  padding: '0',
  cursor: 'pointer',
  outline: 'inherit',
  borderRadius: '8pt',
  width: '16pt',
  height: '16pt',
  fontFamily: 'Verdana',
  fontSize: '8pt',
});

