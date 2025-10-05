module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,html,mdx}"],
  darkMode: "class",
  theme: {
    screens: {
      sm: '640px',   
      md: '768px',    
      lg: '1024px',   
      xl: '1280px',
      '2xl': '1536px'
    },
    extend: {
      colors: {
        // Text Colors
        text: {
          primary: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          white: "var(--text-white)",
          'white-muted': "var(--text-white-muted)"
        },
        // Background Colors
        background: {
          primary: "var(--bg-primary)",
          secondary: "var(--bg-secondary)",
          accent: "var(--bg-accent)"
        },
        // Border Colors
        border: {
          primary: "var(--border-primary)"
        },
        // Component-specific colors
        button: {
          primary: "var(--bg-primary)",
          secondary: "var(--bg-secondary)",
          accent: "var(--bg-accent)"
        },
      },
      fontSize: {
        'sm': 'var(--font-size-sm)',
        'base': 'var(--font-size-base)',
        'lg': 'var(--font-size-lg)',
        'hero': 'var(--font-size-hero)'
      },
      fontWeight: {
        'normal': 'var(--font-weight-normal)',
        'medium': 'var(--font-weight-medium)',
        'bold': 'var(--font-weight-bold)'
      },
      lineHeight: {
        'sm': 'var(--line-height-sm)',
        'base': 'var(--line-height-base)',
        'lg': 'var(--line-height-lg)',
        'hero': 'var(--line-height-hero)'
      },
      spacing: {
        'xs': 'var(--spacing-xs)',
        'sm': 'var(--spacing-sm)',
        'md': 'var(--spacing-md)',
        'lg': 'var(--spacing-lg)',
        'xl': 'var(--spacing-xl)',
        '2xl': 'var(--spacing-2xl)',
        '3xl': 'var(--spacing-3xl)',
        '4xl': 'var(--spacing-4xl)',
        '5xl': 'var(--spacing-5xl)',
        'hero': 'var(--spacing-hero)'
      },
      borderRadius: {
        'md': 'var(--radius-md)',
        'lg': 'var(--radius-lg)'
      },
      borderWidth: {
        'sm': 'var(--border-width-sm)'
      }
    }
  },
  plugins: []
};
