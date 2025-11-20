import React from 'react';
import { AuroraText } from '@/components/ui/aurora-text';
import { cn } from '@/lib/utils';
import { AnimatedShinyText } from '@/components/ui/animated-shiny-text';
import { SparklesText } from '@/components/ui/sparkles-text';
import { Link } from 'react-router-dom';
import { HeroVideoDialog } from '@/components/ui/hero-video-dialog';
import { main } from 'motion/react-client';
import DotGrid from '@/components/ui/DotGrid';

interface HeroSectionProps {
  className?: string;
}

const HeroSection = ({ className }: HeroSectionProps) => {
  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
  };

  const handleCopy = (e: React.ClipboardEvent) => {
    e.preventDefault();
  };

  return (
    <main
      onContextMenu={handleContextMenu}
      onCopy={handleCopy}
      style={{
        userSelect: 'none',
        WebkitUserSelect: 'none',
        MozUserSelect: 'none',
        msUserSelect: 'none',
        cursor: 'default',
      }}
    >
      {/* Sección 1 */}
      <section
        className={`w-full bg-background-primary flex flex-col justify-start items-center min-h-screen ${className || ''}`}
      >
        <div
          className="w-full max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 
  flex flex-col justify-center items-center min-h-screen text-center"
        >
          {/* Main Heading */}
          <h1
            className="text-[31px] sm:text-[43px] md:text-[50px] lg:text-hero 
    font-bold leading-[38px] sm:leading-[53px] md:leading-[61px] lg:leading-hero 
    bg-[linear-gradient(180deg,#ffffff_0%,_#1d1d1d_100%)] 
    bg-clip-text text-transparent"
          >
            Presentamos <AuroraText>Aura</AuroraText>
          </h1>

          <div
            className="flex flex-col gap-spacing-sm justify-center items-center 
    w-full sm:w-[80%] md:w-[70%] lg:w-[42%] mt-4"
          >
            <p className="text-lg font-normal leading-lg text-text-secondary">
              Nos entusiasma lanzar Aura IA para que los usuarios puedan compartir sus opiniones
              sobre sus fortalezas y debilidades. Durante la fase de investigación, Aura será de uso
              gratuito.
            </p>

            {/* Button */}
            <div className="z-10 flex min-h-4 items-center justify-center mt-6">
              <Link
                to="/register"
                className={cn(
                  'group rounded-full border border-black/5 bg-[#020412b7] text-base text-[#686868] transition-all ease-in hover:cursor-pointer select-none active:scale-95 duration-200 dark:border-white/5'
                )}
              >
                <AnimatedShinyText className="inline-flex items-center justify-center px-4 py-1 transition ease-out hover:text-[#CA5CF5] hover:duration-300 dark:hover:text-[#7405B4]">
                  <span className="flex items-center justify-center relative">
                    <img
                      src="/images/img_1.svg"
                      alt="Arrow icon"
                      className="w-[10px] sm:w-[14px] md:w-[17px] lg:w-[20px] h-[10px] sm:h-[14px] md:h-[17px] lg:h-[20px] absolute left-[2px] sm:left-[3px] md:left-[3px] lg:left-spacing-xs"
                    />
                    <span className="ml-[14px] sm:ml-[20px] md:ml-[24px] lg:ml-[28px]">
                      Presentamos Aura IA →
                    </span>
                  </span>
                </AnimatedShinyText>
              </Link>
            </div>
          </div>
        </div>
      </section>
      {/* Sección 2 */}
      <section className="relative w-full min-h-screen flex items-center justify-center overflow-hidden">
        {/* Fondo */}
        <div className="absolute inset-0 z-0">
          <DotGrid
            dotSize={8}
            gap={18}
            baseColor="#020412" // gris oscuro de fondo
            activeColor="#8B3DFF" // morado brillante al pasar el cursor
            proximity={160}
            shockRadius={250}
            shockStrength={5}
            resistance={800}
            returnDuration={1.6}
          />
        </div>

        {/* Difuminado superior */}
        <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-[#020412] to-transparent pointer-events-none" />

        {/* Difuminado inferior */}
        <div className="absolute bottom-0 left-0 w-full h-32 bg-gradient-to-t from-[#020412] to-transparent pointer-events-none" />

        <div
          className="absolute inset-0 z-[1] pointer-events-none"
          style={{
            background:
              'radial-gradient(circle at center, rgba(80,0,150,0.15) 0%, rgba(0,0,0,0.15) 50%)',
            backdropFilter: 'blur(7px)',
          }}
        ></div>

        {/* Contenido principal */}
        <div className="relative z-10 max-w-[1280px] mx-auto px-6 lg:px-8 grid md:grid-cols-2 gap-12 items-center">
          {/* Columna Izquierda */}
          <div className="text-center">
            <h2
              className="text-3xl font-bold leading-snug 
  bg-[linear-gradient(180deg,#ffffff_0%,_#1d1d1d_100%)] 
  bg-clip-text text-transparent"
            >
              Olvídate de las búsquedas rígidas
            </h2>
            <p className="text-lg text-text-secondary leading-relaxed">
              <AuroraText>AURA</AuroraText> te proporciona respuestas claras, organizadas y
              lógicamente coherentes, tal como lo ves en el ejemplo de los departamentos de
              Colombia. Nuestra misión es ser tu aliado más inteligente, transformando cada consulta
              en una experiencia de aprendizaje y productividad.
            </p>
          </div>

          {/* Columna Derecha - Video */}
          <div className="relative flex justify-center">
            <HeroVideoDialog
              className="block dark:hidden rounded-xl overflow-hidden shadow-lg"
              animationStyle="top-in-bottom-out"
              videoSrc="https://www.youtube.com/embed/q8N8Y8qgTE0?si=rB2rruV3osb9J2fn"
              thumbnailSrc="https://i.postimg.cc/DZnP8M7M/Conversaci-n.png"
              thumbnailAlt="Hero Video"
            />
            <HeroVideoDialog
              className="hidden dark:block rounded-xl overflow-hidden shadow-lg"
              animationStyle="top-in-bottom-out"
              videoSrc="https://www.youtube.com/embed/q8N8Y8qgTE0?si=rB2rruV3osb9J2fn"
              thumbnailSrc="https://i.postimg.cc/DZnP8M7M/Conversaci-n.png"
              thumbnailAlt="Hero Video"
            />
          </div>
        </div>
      </section>

      {/* Sección 3 */}
      <section className="w-full bg-background-primary min-h-screen flex items-center justify-center">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white">Características</h2>
          <p className="mt-4 text-lg text-text-secondary">No se.</p>
        </div>
      </section>
    </main>
  );
};

export default HeroSection;
