import React from "react";
import "./Contact.css";
import { FaLinkedin, FaGithub } from "react-icons/fa";

const teamMembers = [
  {
    name: "Abhinav Sharma",
    role: "Frontend Developer",
    linkedin: "https://www.linkedin.com/in/abhinav-sharma",
    github: "https://github.com/abhinav-sharma",
  },
  {
    name: "Lakshay Asnani",
    role: "Backend Developer",
    linkedin: "https://www.linkedin.com/in/teammate1",
    github: "https://github.com/teammate1",
  },
  {
    name: " Pranavi Gupta",
    role: " Designer",
    linkedin: "https://www.linkedin.com/in/teammate2",
    github: "https://github.com/teammate2",
  },
  {
    name: "Aryan ",
    role: "Backend Developer",
    linkedin: "https://www.linkedin.com/in/teammate2",
    github: "https://github.com/teammate2",
  },
];

const Contact = () => {
  return (
    <div className="contact-container">
      <h1 className="contact-title">Get in Touch</h1>
      <p className="contact-description">
        Have questions or want to collaborate? Send us a message and connect
        with our team members below.
      </p>

      <form className="contact-form">
        <input type="text" placeholder="Your Name" required />
        <input type="email" placeholder="Your Email" required />
        <textarea placeholder="Your Message" rows="5" required></textarea>
        <button type="submit">Send Message</button>
      </form>

      <h2 className="team-title">Meet the Team</h2>
      <div className="team-cards">
        {teamMembers.map((member, idx) => (
          <div key={idx} className="team-card">
            <h3>{member.name}</h3>
            <p>{member.role}</p>
            <div className="social-links">
              <a
                href={member.linkedin}
                target="_blank"
                rel="noopener noreferrer"
              >
                <FaLinkedin size={25} color="#0A66C2" />
              </a>
              <a href={member.github} target="_blank" rel="noopener noreferrer">
                <FaGithub size={25} color="#fff" />
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Contact;
