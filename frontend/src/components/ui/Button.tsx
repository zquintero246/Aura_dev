import React from 'react';
import { cva, VariantProps } from 'class-variance-authority';
import { twMerge } from 'tailwind-merge';

const buttonClasses = cva(
  'inline-flex items-center justify-center font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 active:scale-95',
  {
    variants: {
      variant: {
        primary: 'bg-button-primary text-text-white hover:bg-opacity-90 focus:ring-gray-500',
        secondary: 'bg-button-secondary text-text-white hover:bg-opacity-90 focus:ring-gray-500',
        accent: 'bg-button-accent text-text-primary hover:bg-opacity-90 focus:ring-purple-500',
      },
      size: {
        small: 'text-sm px-3 py-1.5',
        medium: 'text-base px-4 py-2',
        large: 'text-lg px-6 py-3',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'medium',
    },
  }
);

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonClasses> {
  // Required parameters with defaults
  text?: string;
  text_font_size?: string;
  text_font_family?: string;
  text_font_weight?: string;
  text_line_height?: string;
  text_text_align?: 'left' | 'center' | 'right' | 'justify';
  text_color?: string;
  fill_background_color?: string;
  border_border_radius?: string;

  // Optional parameters (no defaults)
  border_border?: string;
  layout_width?: string;
  padding?: string;
  position?: string;
  layout_gap?: string;

  // Standard React props
  variant?: 'primary' | 'secondary' | 'accent';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  className?: string;
  children?: React.ReactNode;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  type?: 'button' | 'submit' | 'reset';
}

const Button = ({
  // Required parameters with defaults
  text = 'Ingresar',
  text_font_size = 'text-base',
  text_font_family = 'Inter',
  text_font_weight = 'font-medium',
  text_line_height = 'leading-base',
  text_text_align = 'left',
  text_color = 'text-text-primary',
  fill_background_color = 'bg-button-accent',
  border_border_radius = 'rounded-md',

  // Optional parameters (no defaults)
  border_border,
  layout_width,
  padding,
  position,
  layout_gap,

  // Standard React props
  variant,
  size,
  disabled = false,
  className,
  children,
  onClick,
  type = 'button',
  ...props
}: ButtonProps) => {
  // Safe validation for optional parameters
  const hasValidBorder =
    border_border && typeof border_border === 'string' && border_border.trim() !== '';
  const hasValidWidth =
    layout_width && typeof layout_width === 'string' && layout_width.trim() !== '';
  const hasValidPadding = padding && typeof padding === 'string' && padding.trim() !== '';
  const hasValidPosition = position && typeof position === 'string' && position.trim() !== '';
  const hasValidGap = layout_gap && typeof layout_gap === 'string' && layout_gap.trim() !== '';

  // Build optional Tailwind classes
  const optionalClasses = [
    hasValidBorder ? `border-[${border_border}]` : '',
    hasValidWidth ? `w-[${layout_width}]` : '',
    hasValidPadding ? `p-[${padding}]` : '',
    hasValidPosition ? position : '',
    hasValidGap ? `gap-[${layout_gap}]` : '',
  ]
    .filter(Boolean)
    .join(' ');

  // Build required styles
  const requiredClasses = [
    text_font_size,
    `font-[${text_font_family}]`,
    text_font_weight,
    text_line_height,
    `text-${text_text_align}`,
    text_color,
    fill_background_color,
    border_border_radius,
  ].join(' ');

  // Safe click handler
  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    if (disabled) return;
    if (typeof onClick === 'function') {
      onClick(event);
    }
  };

  return (
    <button
      type={type}
      disabled={disabled}
      onClick={handleClick}
      className={twMerge(
        buttonClasses({ variant, size }),
        requiredClasses,
        optionalClasses,
        className
      )}
      aria-disabled={disabled}
      {...props}
    >
      {children || text}
    </button>
  );
};

export default Button;
