import React from 'react';
import { cva, VariantProps } from 'class-variance-authority';
import { twMerge } from 'tailwind-merge';

const dividerClasses = cva('border-0', {
  variants: {
    orientation: {
      horizontal: 'w-full h-px',
      vertical: 'h-full w-px',
    },
    variant: {
      solid: 'bg-current',
      dashed: 'border-dashed border-t border-current bg-transparent',
      dotted: 'border-dotted border-t border-current bg-transparent',
    },
    color: {
      default: 'text-line-background',
      muted: 'text-text-muted',
      primary: 'text-background-accent',
    },
  },
  defaultVariants: {
    orientation: 'horizontal',
    variant: 'solid',
    color: 'default',
  },
});

interface DividerProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof dividerClasses> {
  // Custom props
  orientation?: 'horizontal' | 'vertical';
  variant?: 'solid' | 'dashed' | 'dotted';
  color?: 'default' | 'muted' | 'primary';
  thickness?: string;
  length?: string;
  className?: string;

  // Text divider props
  text?: string;
  textClassName?: string;
}

const Divider = ({
  orientation = 'horizontal',
  variant = 'solid',
  color = 'default',
  thickness,
  length,
  text,
  textClassName,
  className,
  ...props
}: DividerProps) => {
  // Build custom thickness and length classes
  const customClasses = [
    thickness && orientation === 'horizontal' ? `h-[${thickness}]` : '',
    thickness && orientation === 'vertical' ? `w-[${thickness}]` : '',
    length && orientation === 'horizontal' ? `w-[${length}]` : '',
    length && orientation === 'vertical' ? `h-[${length}]` : '',
  ]
    .filter(Boolean)
    .join(' ');

  // If text is provided, render as text divider
  if (text) {
    return (
      <div
        className={twMerge('flex items-center gap-4 w-full', className)}
        role="separator"
        aria-label={text}
        {...props}
      >
        <div
          className={twMerge(
            dividerClasses({
              orientation: 'horizontal',
              variant,
              color,
            }),
            'flex-1',
            customClasses
          )}
        />
        <span className={twMerge('text-sm text-text-muted whitespace-nowrap', textClassName)}>
          {text}
        </span>
        <div
          className={twMerge(
            dividerClasses({
              orientation: 'horizontal',
              variant,
              color,
            }),
            'flex-1',
            customClasses
          )}
        />
      </div>
    );
  }

  // Regular divider without text
  return (
    <div
      className={twMerge(dividerClasses({ orientation, variant, color }), customClasses, className)}
      role="separator"
      {...props}
    />
  );
};

export default Divider;
