import React from 'react';
import { cva, VariantProps } from 'class-variance-authority';
import { twMerge } from 'tailwind-merge';

const buttonClasses = cva(
  'inline-flex items-center justify-center font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed',
  {
    variants: {
      variant: {
        primary: 'hover:opacity-90 focus:ring-purple-500',
        secondary: 'bg-gray-200 text-gray-800 hover:bg-gray-300 focus:ring-gray-500',
        outline: 'border-2 bg-transparent hover:bg-opacity-10 focus:ring-purple-500',
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
  margin?: string;
  position?: string;
  layout_gap?: string;

  // Standard React props
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  className?: string;
  children?: React.ReactNode;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  type?: 'button' | 'submit' | 'reset';
}

const Button = ({
  // Required parameters with defaults
  text = 'Verificar correo',
  text_font_size = '15',
  text_font_family = 'Inter',
  text_font_weight = '700',
  text_line_height = '19px',
  text_text_align = 'left',
  text_color = '#1d1d1d',
  fill_background_color = '#ca5cf5',
  border_border_radius = '10px',

  // Optional parameters (no defaults)
  border_border,
  layout_width,
  padding,
  margin,
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
  const hasValidMargin = margin && typeof margin === 'string' && margin.trim() !== '';
  const hasValidPosition = position && typeof position === 'string' && position.trim() !== '';
  const hasValidGap = layout_gap && typeof layout_gap === 'string' && layout_gap.trim() !== '';

  // Build optional Tailwind classes
  const optionalClasses = [
    hasValidWidth ? `w-[${layout_width}]` : '',
    hasValidPadding ? `p-[${padding}]` : '',
    hasValidMargin ? `m-[${margin}]` : '',
    hasValidPosition ? position : '',
    hasValidGap ? `gap-[${layout_gap}]` : '',
  ]
    .filter(Boolean)
    .join(' ');

  // Map style values to Tailwind classes or use hardcoded values
  const getFontSize = (size: string) => {
    if (size === '15') return 'text-sm'; // matches text-sm: 15px
    return `text-[${size}px]`;
  };

  const getFontWeight = (weight: string) => {
    if (weight === '700') return 'font-bold'; // matches font-bold: 700
    return `font-[${weight}]`;
  };

  const getLineHeight = (height: string) => {
    if (height === '19px') return 'leading-md'; // matches leading-md: 19px
    return `leading-[${height}]`;
  };

  const getTextColor = (color: string) => {
    if (color === '#1d1d1d') return 'text-text-secondary'; // matches text-text-secondary
    return `text-[${color}]`;
  };

  const getBackgroundColor = (color: string) => {
    if (color === '#ca5cf5') return 'bg-background-accent'; // matches bg-background-accent
    return `bg-[${color}]`;
  };

  const getBorderRadius = (radius: string) => {
    if (radius === '10px') return 'rounded-md'; // matches rounded-md: 10px
    return `rounded-[${radius}]`;
  };

  const getTextAlign = (align: string) => {
    switch (align) {
      case 'left':
        return 'text-left';
      case 'center':
        return 'text-center';
      case 'right':
        return 'text-right';
      case 'justify':
        return 'text-justify';
      default:
        return 'text-left';
    }
  };

  // Build style classes
  const styleClasses = [
    getFontSize(text_font_size),
    `font-[${text_font_family}]`,
    getFontWeight(text_font_weight),
    getLineHeight(text_line_height),
    getTextAlign(text_text_align),
    getTextColor(text_color),
    getBackgroundColor(fill_background_color),
    getBorderRadius(border_border_radius),
    hasValidBorder ? `border-[${border_border}]` : '',
  ]
    .filter(Boolean)
    .join(' ');

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
        styleClasses,
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
