import React from 'react';
import { Helmet } from 'react-helmet';
import Header from '../../components/common/Header';
import HeroSection from './HeroSection';
import GradientBlinds from '../../components/GradientBlinds';
import { GridBeams } from '../../components/grid-beams';

const Home = () => {
  return (
    <>
      <Helmet>
        <title>Aura AI</title>
        <meta
          name="description"
          content="Join Aura AI research phase with free access to cutting-edge AI conversations. Share feedback on strengths & weaknesses to shape the future of AI technology."
        />
        <meta
          property="og:title"
          content="Aura AI - AI Research Platform | Free Beta Access & Feedback"
        />
        <meta
          property="og:description"
          content="Join Aura AI research phase with free access to cutting-edge AI conversations. Share feedback on strengths & weaknesses to shape the future of AI technology."
        />
      </Helmet>

      <main className="relative w-full min-h-screen overflow-hidden">
        {/* Fondo animado */}
        <div className="fixed inset-0 -z-10">
          <GridBeams>
            {/* Este div vacío mantiene la estructura pero ahora ocupa toda la pantalla */}
            <div className="w-full h-screen" />
          </GridBeams>
        </div>

        {/* Contenido */}
        <div className="relative z-10">
          <Header />
          <HeroSection />
        </div>
      </main>
    </>
  );
};

export default Home;
