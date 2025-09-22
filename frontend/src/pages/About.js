import React from "react";
import "./About.css";
import Tilt from "react-parallax-tilt";

const About = () => {
  return (
    <div className="about-container">
      <h1 className="about-title">About PaperPirates</h1>
      <p className="about-description">
        PaperPirates is a platform designed to make learning interactive and fun
        for students. Upload your PDFs, get instant insights, and collaborate
        efficiently with AI-powered tools.
      </p>

      <div className="about-cards">
        <Tilt glareEnable={true} glareMaxOpacity={0.3} scale={1.05}>
          <div className="about-card">
            <h3>Learn Smarter</h3>
            <p>
              Get AI-assisted insights from your study materials to maximize
              learning efficiency.
            </p>
          </div>
        </Tilt>

        <Tilt glareEnable={true} glareMaxOpacity={0.3} scale={1.05}>
          <div className="about-card">
            <h3>Tech-Enabled</h3>
            <p>
              Seamlessly upload PDFs and access interactive tools designed for
              modern students.
            </p>
          </div>
        </Tilt>

        <Tilt glareEnable={true} glareMaxOpacity={0.3} scale={1.05}>
          <div className="about-card">
            <h3>Collaborate</h3>
            <p>
              Share insights and work together with peers using our AI-powered
              features.
            </p>
          </div>
        </Tilt>
      </div>
    </div>
  );
};

export default About;
