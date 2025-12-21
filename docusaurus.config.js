// @ts-check
// `@ts-check` enables ts-autocomplete

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Humanoid Robotics Book',
  tagline: 'A Comprehensive Guide to ROS 2, Simulation, AI, and VLA Integration',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  //url: 'https://farooq0319ahmed.github.io',
  url: 'https://hackathon-ai-humanoid-robotics-textbook.vercel.app',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  //baseUrl: '/Hackathon-AI-Humanoid-Robotics-Textbook/',
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  //organizationName: 'farooq0319ahmed', // Usually your GitHub org/user name.
  //projectName: 'Hackathon-AI-Humanoid-Robotics-Textbook', // Usually your repo name.

  //onBrokenLinks: 'throw',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/docs', // Serve docs under /docs path
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/farooq0319ahmed/Hackathon-AI-Humanoid-Robotics-Textbook/tree/main/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Humanoid Robotics Book',
        logo: {
          alt: 'Robot Logo',
          src: 'img/robot-logo.svg',
          href: '/docs/intro',  // Link logo to intro page
        },
        items: [
          //{
            //type: 'doc',
            //docId: 'intro',
            //position: 'left',
            //label: 'Home',
          //},
          {
            to: '/',
            label: 'Home',
            position: 'left',
          },
          {
            to: '/docs/intro',
            label: 'Intro',
            position: 'left',
          },
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Modules',
          },
          {
            href: 'https://github.com/farooq0319ahmed/Hackathon-AI-Humanoid-Robotics-Textbook',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'ROS 2 Fundamentals',
                to: '/docs/modules/ros-nervous-system',
              },
              {
                label: 'Simulation',
                to: '/docs/modules/digital-twin',
              },
              {
                label: 'AI Perception',
                to: '/docs/modules/ai-brain',
              },
              {
                label: 'VLA & Capstone',
                to: '/docs/modules/vla-capstone',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'ROS Answers',
                href: 'https://answers.ros.org/',
              },
              {
                label: 'Docusaurus',
                href: 'https://docusaurus.io/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/farooq0319ahmed/Hackathon-AI-Humanoid-Robotics-Textbook',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Humanoid Robotics Book. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;