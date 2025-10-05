import React, { useState } from 'react';
import Button from '../ui/Button';

interface HeaderProps {
  className?: string;
}

const Header = ({ className }: HeaderProps) => {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 w-full 
    bg-[linear-gradient(180deg,rgba(2,4,18,1.5)_0%,rgba(29,29,29,0)_100%)] 
    backdrop-blur-md
    pt-[15px] sm:pt-[22px] md:pt-[26px] lg:pt-[30px] 
    pr-[7px] sm:pr-[10px] md:pr-[12px] lg:pr-[14px] 
    pb-[15px] sm:pb-[22px] md:pb-[26px] lg:pb-[30px] 
    pl-[7px] sm:pl-[10px] md:pl-[12px] lg:pl-[14px] 
    ${className || ''}`}
    >
      <div className="w-full max-w-[1440px] mx-auto">
        <div className="flex flex-row justify-between items-center w-full mb-[14px] sm:mb-[20px] md:mb-[24px] lg:mb-[28px] ml-[6px] sm:ml-[8px] md:ml-[10px] lg:ml-[12px]">
          {/* Logo */}
          <div className="flex items-center">
            <img
              src="/images/img_header_logo.png"
              alt="Aura CosoGPT Logo"
              className="w-[45px] sm:w-[60px] md:w-[75px] lg:w-[90px] h-[16px] sm:h-[21px] md:h-[27px] lg:h-[32px]"
            />
          </div>

          {/* Hamburger Menu Icon (Mobile only) */}
          <button
            className="block lg:hidden p-2"
            aria-label="Open menu"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <svg
              className="w-6 h-6 text-text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>

          {/* Navigation Buttons */}
          <nav
            className={`${menuOpen ? 'block' : 'hidden'} lg:block absolute lg:relative top-full lg:top-auto left-0 lg:left-auto w-full lg:w-auto bg-background-secondary lg:bg-transparent p-4 lg:p-0 z-50`}
          >
            <div className="flex flex-col lg:flex-row gap-[7px] sm:gap-[10px] md:gap-[12px] lg:gap-[14px] justify-center items-center w-full lg:w-auto">
              <Button
                text="Ingresar"
                text_font_size="text-base"
                text_font_family="Inter"
                text_font_weight="font-medium"
                text_line_height="leading-base"
                text_text_align="left"
                text_color="text-text-primary"
                fill_background_color="bg-button-accent"
                border_border_radius="rounded-md"
                padding="pt-[2px] sm:pt-[3px] md:pt-[3px] lg:pt-[4px] pr-[11px] sm:pr-[16px] md:pr-[19px] lg:pr-[22px] pb-[2px] sm:pb-[3px] md:pb-[3px] lg:pb-[4px] pl-[11px] sm:pl-[16px] md:pl-[19px] lg:pl-[22px]"
                layout_width="auto"
                className="w-full lg:w-auto"
              />
              <Button
                text="Registrarse"
                text_font_size="text-base"
                text_font_family="Inter"
                text_font_weight="font-medium"
                text_line_height="leading-base"
                text_text_align="left"
                text_color="text-text-white"
                fill_background_color="bg-button-primary"
                border_border_radius="rounded-md"
                padding="pt-[2px] sm:pt-[3px] md:pt-[3px] lg:pt-[4px] pr-[5px] sm:pr-[7px] md:pr-[8px] lg:pr-[10px] pb-[2px] sm:pb-[3px] md:pb-[3px] lg:pb-[4px] pl-[5px] sm:pl-[7px] md:pl-[8px] lg:pl-[10px]"
                layout_width="auto"
                className="w-full lg:w-auto"
              />
            </div>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
