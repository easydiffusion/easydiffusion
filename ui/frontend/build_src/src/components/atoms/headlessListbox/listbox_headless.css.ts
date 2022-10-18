import { style } from '@vanilla-extract/css';
import { calc } from '@vanilla-extract/css-utils';
import { vars } from '@styles/theme/index.css';

export const ListboxHeadless = style({
  position: 'relative',
});

export const ListboxHeadlessLabel = style({
  marginRight: vars.spacing.small,
});

export const ListboxHeadlessButton = style({
  position: 'relative',
  backgroundColor: vars.backgroundLight,
  border: 'none',
  marginTop: vars.spacing.min,
  padding: `${vars.spacing.min} ${vars.spacing.small}`,
  borderRadius: vars.trim.smallBorderRadius,
});

export const ListboxHeadlessOptions = style({
  position: 'absolute',
  backgroundColor: vars.backgroundLight,
  zIndex: 1,
  bottom: calc(vars.spacing.small).negate().toString(),
  transform: `translateY(100%)`,
  borderRadius: vars.trim.smallBorderRadius,
  padding: `${vars.spacing.small} 0`,
});

export const ListboxHeadlessOptionItem = style({
  padding: `0 ${vars.spacing.small}`,

  selectors: {
    [`&[data-headlessui-state="active"]`]: {
      backgroundColor: vars.backgroundDark,
    },
    [`&[data-headlessui-state="selected"]`]: {
      backgroundColor: vars.backgroundMain,
    },
    [`&[data-headlessui-state="active selected"]`]: {
      backgroundColor: vars.backgroundDark,
    }
  }

});

export const ListBoxIcon = style({
  marginLeft: vars.spacing.small,
});

