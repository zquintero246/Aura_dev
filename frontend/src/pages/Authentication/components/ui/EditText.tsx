import React, { useState, forwardRef } from 'react';
import { cva, VariantProps } from 'class-variance-authority';
import { twMerge } from 'tailwind-merge';

const inputClasses = cva(
  'w-full transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed',
  {
    variants: {
      variant: {
        default: 'focus:ring-purple-500',
        error: 'border-red-500 focus:ring-red-500',
        success: 'border-green-500 focus:ring-green-500',
      },
      size: {
        small: 'text-sm px-3 py-2',
        medium: 'text-base px-4 py-3',
        large: 'text-lg px-5 py-4',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'medium',
    },
  }
);

interface EditTextProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'>,
    VariantProps<typeof inputClasses> {
  // Required parameters with defaults
  fill_background_color?: string;
  border_border?: string;
  border_border_radius?: string;

  // Optional parameters (no defaults)
  layout_width?: string;
  padding?: string;
  margin?: string;
  position?: string;

  // Standard React props
  variant?: 'default' | 'error' | 'success';
  size?: 'small' | 'medium' | 'large';
  label?: string;
  error?: string;
  helperText?: string;
  className?: string;
  containerClassName?: string;
}

const EditText = forwardRef<HTMLInputElement, EditTextProps>(
  (
    {
      // Required parameters with defaults
      fill_background_color = '#2d2d2d',
      border_border = '1 solid #686868',
      border_border_radius = '10px',

      // Optional parameters (no defaults)
      layout_width,
      padding,
      margin,
      position,

      // Standard React props
      variant,
      size,
      label,
      error,
      helperText,
      className,
      containerClassName,
      disabled = false,
      type = 'text',
      ...props
    },
    ref
  ) => {
    const [isFocused, setIsFocused] = useState(false);

    // Safe validation for optional parameters
    const hasValidWidth =
      layout_width && typeof layout_width === 'string' && layout_width.trim() !== '';
    const hasValidPadding = padding && typeof padding === 'string' && padding.trim() !== '';
    const hasValidMargin = margin && typeof margin === 'string' && margin.trim() !== '';
    const hasValidPosition = position && typeof position === 'string' && position.trim() !== '';

    // Build optional Tailwind classes
    const optionalClasses = [
      hasValidWidth ? `w-[${layout_width}]` : '',
      hasValidPadding ? `p-[${padding}]` : '',
      hasValidMargin ? `m-[${margin}]` : '',
      hasValidPosition ? position : '',
    ]
      .filter(Boolean)
      .join(' ');

    // Map style values to Tailwind classes or use hardcoded values
    const getBackgroundColor = (color: string) => {
      if (color === '#2d2d2d') return 'bg-input-background'; // matches bg-input-background
      return `bg-[${color}]`;
    };

    const getBorder = (border: string) => {
      if (border === '1 solid #686868') return 'border border-input-border'; // matches border-input-border
      // Parse border string: "width style color"
      const borderParts = border.split(' ');
      if (borderParts.length >= 3) {
        const width = borderParts[0];
        const style = borderParts[1];
        const color = borderParts.slice(2).join(' ');
        return `border-[${width}px] border-${style} border-[${color}]`;
      }
      return `border-[${border}]`;
    };

    const getBorderRadius = (radius: string) => {
      if (radius === '10px') return 'rounded-md'; // matches rounded-md: 10px
      return `rounded-[${radius}]`;
    };

    // Build style classes
    const styleClasses = [
      getBackgroundColor(fill_background_color),
      getBorder(border_border),
      getBorderRadius(border_border_radius),
      'text-text-inverse', // White text for dark background
    ]
      .filter(Boolean)
      .join(' ');

    // Handle focus events
    const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
      setIsFocused(true);
      if (props.onFocus) {
        props.onFocus(e);
      }
    };

    const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
      setIsFocused(false);
      if (props.onBlur) {
        props.onBlur(e);
      }
    };

    // Determine variant based on error state
    const inputVariant = error ? 'error' : variant;

    return (
      <div className={twMerge('flex flex-col gap-1', containerClassName)}>
        {label && <label className="text-sm font-medium text-text-inverse mb-1">{label}</label>}

        <input
          ref={ref}
          type={type}
          disabled={disabled}
          onFocus={handleFocus}
          onBlur={handleBlur}
          className={twMerge(
            inputClasses({ variant: inputVariant, size }),
            styleClasses,
            optionalClasses,
            isFocused && 'ring-2 ring-purple-500 ring-offset-2 ring-offset-background-primary',
            error && 'border-red-500',
            className
          )}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={
            error ? `${props.id}-error` : helperText ? `${props.id}-helper` : undefined
          }
          {...props}
        />

        {error && (
          <span id={`${props.id}-error`} className="text-xs text-red-500 mt-1" role="alert">
            {error}
          </span>
        )}

        {helperText && !error && (
          <span id={`${props.id}-helper`} className="text-xs text-text-muted mt-1">
            {helperText}
          </span>
        )}
      </div>
    );
  }
);

EditText.displayName = 'EditText';

export default EditText;
