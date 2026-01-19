import { createSignal, onMount, For } from 'solid-js';

// Course data
const courses = [
  {
    id: 'quantum',
    title: 'Quantum Computing',
    subtitle: 'Master the Future of Computing',
    description: 'Learn quantum mechanics, qubits, superposition, and entanglement. Build quantum circuits and implement famous algorithms like Grover\'s and Shor\'s.',
    icon: '⚛️',
    gradient: 'from-purple-500 via-violet-500 to-fuchsia-500',
    cardClass: 'course-card-quantum',
    glowClass: 'glow-quantum',
    stats: [
      { label: 'Modules', value: '7' },
      { label: 'Lessons', value: '34+' },
      { label: 'Visualizations', value: '15+' },
    ],
    modules: [
      { name: 'Quantum Fundamentals', icon: '🔬', lessons: 5 },
      { name: 'Quantum Gates', icon: '🚪', lessons: 7 },
      { name: 'Quantum Algorithms', icon: '🧮', lessons: 6 },
      { name: 'Quantum Hardware', icon: '🖥️', lessons: 5 },
      { name: 'Variational Algorithms', icon: '🔄', lessons: 4 },
      { name: 'Quantum ML', icon: '🤖', lessons: 4 },
      { name: 'Advanced Topics', icon: '🚀', lessons: 3 },
    ],
    url: 'http://localhost:5173',
    accentColor: 'purple',
  },
  {
    id: 'aiml',
    title: 'AI & Machine Learning',
    subtitle: 'From Fundamentals to Production',
    description: 'Comprehensive coverage of machine learning, deep learning, NLP, computer vision, and generative AI. Hands-on demos and interactive visualizations.',
    icon: '🧠',
    gradient: 'from-indigo-500 via-blue-500 to-cyan-500',
    cardClass: 'course-card-aiml',
    glowClass: 'glow-aiml',
    stats: [
      { label: 'Modules', value: '10' },
      { label: 'Lessons', value: '30+' },
      { label: 'Demos', value: '8+' },
    ],
    modules: [
      { name: 'Foundations', icon: '🎯', lessons: 4 },
      { name: 'Machine Learning', icon: '📊', lessons: 6 },
      { name: 'Deep Learning', icon: '🧠', lessons: 5 },
      { name: 'NLP', icon: '📝', lessons: 3 },
      { name: 'Computer Vision', icon: '👁️', lessons: 2 },
      { name: 'Time Series', icon: '📈', lessons: 2 },
      { name: 'Generative AI', icon: '✨', lessons: 6 },
      { name: 'MLOps', icon: '⚙️', lessons: 2 },
    ],
    url: 'http://localhost:5174',
    accentColor: 'indigo',
  },
];

const features = [
  {
    icon: '🎮',
    title: 'Interactive Learning',
    description: 'Hands-on visualizations and experiments that make complex concepts intuitive',
  },
  {
    icon: '🔬',
    title: 'Deep Technical Content',
    description: 'From fundamentals to advanced topics with real-world applications',
  },
  {
    icon: '⚡',
    title: 'Practice & Apply',
    description: 'Code playgrounds, quizzes, and capstone projects to solidify your knowledge',
  },
  {
    icon: '🌐',
    title: 'Industry Ready',
    description: 'Learn production-level tools and best practices used by top companies',
  },
];

export default function App() {
  const [mounted, setMounted] = createSignal(false);
  const [hoveredCourse, setHoveredCourse] = createSignal<string | null>(null);

  onMount(() => {
    setMounted(true);
  });

  return (
    <div class="min-h-screen relative overflow-hidden">
      {/* Animated Background */}
      <div class="fixed inset-0 hero-bg" />

      {/* Grid Pattern Overlay */}
      <div class="fixed inset-0 grid-pattern opacity-50" />

      {/* Floating Orbs */}
      <div class="orb orb-quantum w-96 h-96 -top-20 -left-20 animate-float" />
      <div class="orb orb-aiml w-80 h-80 top-1/3 -right-20 animate-float delay-200" />
      <div class="orb orb-quantum w-64 h-64 bottom-20 left-1/4 animate-float delay-400" />

      {/* Content */}
      <div class="relative z-10">
        {/* Header */}
        <header class="px-6 py-6">
          <div class="max-w-7xl mx-auto flex items-center justify-between">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center text-xl shadow-lg">
                📚
              </div>
              <span class="text-xl font-bold text-white">Training Hub</span>
            </div>
            <nav class="hidden md:flex items-center gap-8">
              <a href="#courses" class="text-gray-300 hover:text-white transition-colors">Courses</a>
              <a href="#features" class="text-gray-300 hover:text-white transition-colors">Features</a>
              <a href="#about" class="text-gray-300 hover:text-white transition-colors">About</a>
            </nav>
          </div>
        </header>

        {/* Hero Section */}
        <section class="px-6 pt-16 pb-24">
          <div class="max-w-7xl mx-auto text-center">
            <div
              class={`transition-all duration-1000 ${mounted() ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}
            >
              <div class="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm border border-white/20 rounded-full px-4 py-2 mb-8">
                <span class="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span class="text-sm text-gray-300">Now with interactive visualizations</span>
              </div>

              <h1 class="text-5xl md:text-7xl font-bold mb-6">
                <span class="text-white">Master </span>
                <span class="gradient-text">Cutting-Edge</span>
                <br />
                <span class="text-white">Computing</span>
              </h1>

              <p class="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto mb-12 text-balance">
                Interactive courses on Quantum Computing and AI/ML.
                Learn by doing with real-time visualizations, hands-on experiments, and production-ready skills.
              </p>

              <div class="flex flex-wrap gap-4 justify-center">
                <a href="#courses" class="btn-primary text-lg">
                  Explore Courses →
                </a>
                <a href="#features" class="btn-secondary text-lg">
                  See Features
                </a>
              </div>
            </div>

            {/* Stats */}
            <div
              class={`mt-20 grid grid-cols-2 md:grid-cols-4 gap-8 transition-all duration-1000 delay-300 ${mounted() ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}
            >
              {[
                { value: '60+', label: 'Lessons' },
                { value: '17', label: 'Modules' },
                { value: '20+', label: 'Visualizations' },
                { value: '8+', label: 'Interactive Demos' },
              ].map(stat => (
                <div class="text-center">
                  <div class="text-4xl md:text-5xl font-bold gradient-text mb-2">{stat.value}</div>
                  <div class="text-gray-400">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Courses Section */}
        <section id="courses" class="px-6 py-24">
          <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
              <h2 class="text-4xl md:text-5xl font-bold text-white mb-4">Choose Your Path</h2>
              <p class="text-xl text-gray-400 max-w-2xl mx-auto">
                Two comprehensive learning tracks designed to take you from beginner to expert
              </p>
            </div>

            <div class="grid lg:grid-cols-2 gap-8">
              <For each={courses}>
                {(course) => (
                  <div
                    class={`card ${course.cardClass} p-8 relative overflow-hidden group`}
                    onMouseEnter={() => setHoveredCourse(course.id)}
                    onMouseLeave={() => setHoveredCourse(null)}
                  >
                    {/* Decorative gradient blob */}
                    <div
                      class={`absolute -top-20 -right-20 w-60 h-60 bg-gradient-to-br ${course.gradient} rounded-full opacity-20 blur-3xl transition-all duration-500 group-hover:opacity-40 group-hover:scale-125`}
                    />

                    <div class="relative z-10">
                      {/* Header */}
                      <div class="flex items-start justify-between mb-6">
                        <div
                          class={`w-16 h-16 bg-gradient-to-br ${course.gradient} rounded-2xl flex items-center justify-center text-3xl shadow-lg group-hover:scale-110 transition-transform duration-300`}
                        >
                          {course.icon}
                        </div>
                        <div class="flex gap-2">
                          <For each={course.stats}>
                            {(stat) => (
                              <div class="bg-white/10 backdrop-blur-sm rounded-lg px-3 py-1.5 text-center">
                                <div class="text-lg font-bold text-white">{stat.value}</div>
                                <div class="text-xs text-gray-400">{stat.label}</div>
                              </div>
                            )}
                          </For>
                        </div>
                      </div>

                      {/* Title & Description */}
                      <h3 class="text-2xl md:text-3xl font-bold text-white mb-2">{course.title}</h3>
                      <p class={`text-lg font-medium bg-gradient-to-r ${course.gradient} bg-clip-text text-transparent mb-4`}>
                        {course.subtitle}
                      </p>
                      <p class="text-gray-400 mb-8 leading-relaxed">{course.description}</p>

                      {/* Modules Preview */}
                      <div class="mb-8">
                        <h4 class="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
                          Modules Included
                        </h4>
                        <div class="flex flex-wrap gap-2">
                          <For each={course.modules.slice(0, 5)}>
                            {(module) => (
                              <div class="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm flex items-center gap-2">
                                <span>{module.icon}</span>
                                <span class="text-gray-300">{module.name}</span>
                              </div>
                            )}
                          </For>
                          {course.modules.length > 5 && (
                            <div class="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-400">
                              +{course.modules.length - 5} more
                            </div>
                          )}
                        </div>
                      </div>

                      {/* CTA */}
                      <a
                        href={course.url}
                        class={`inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r ${course.gradient} text-white font-semibold rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105`}
                      >
                        Start Learning
                        <svg class="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3" />
                        </svg>
                      </a>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" class="px-6 py-24">
          <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
              <h2 class="text-4xl md:text-5xl font-bold text-white mb-4">Why Learn Here?</h2>
              <p class="text-xl text-gray-400 max-w-2xl mx-auto">
                Our courses are designed for maximum learning effectiveness
              </p>
            </div>

            <div class="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              <For each={features}>
                {(feature, index) => (
                  <div
                    class={`card glass p-6 text-center animate-fade-in`}
                    style={{ 'animation-delay': `${index() * 0.1}s` }}
                  >
                    <div class="w-16 h-16 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center text-3xl mx-auto mb-4 border border-white/10">
                      {feature.icon}
                    </div>
                    <h3 class="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                    <p class="text-gray-400 text-sm">{feature.description}</p>
                  </div>
                )}
              </For>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section class="px-6 py-24">
          <div class="max-w-4xl mx-auto">
            <div class="card glass p-12 text-center relative overflow-hidden">
              {/* Decorative elements */}
              <div class="absolute -top-10 -left-10 w-40 h-40 bg-gradient-to-br from-purple-500 to-indigo-500 rounded-full opacity-20 blur-3xl" />
              <div class="absolute -bottom-10 -right-10 w-40 h-40 bg-gradient-to-br from-indigo-500 to-cyan-500 rounded-full opacity-20 blur-3xl" />

              <div class="relative z-10">
                <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
                  Ready to Start Your Journey?
                </h2>
                <p class="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
                  Choose your path and begin learning with interactive content and hands-on practice
                </p>
                <div class="flex flex-wrap gap-4 justify-center">
                  <a href={courses[0].url} class="btn-primary">
                    ⚛️ Start Quantum Computing
                  </a>
                  <a href={courses[1].url} class="btn-primary">
                    🧠 Start AI/ML
                  </a>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer id="about" class="px-6 py-12 border-t border-white/10">
          <div class="max-w-7xl mx-auto">
            <div class="flex flex-col md:flex-row items-center justify-between gap-6">
              <div class="flex items-center gap-3">
                <div class="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center text-xl shadow-lg">
                  📚
                </div>
                <div>
                  <span class="text-xl font-bold text-white">Training Hub</span>
                  <p class="text-sm text-gray-400">Master Quantum Computing & AI/ML</p>
                </div>
              </div>

              <div class="flex gap-8 text-gray-400">
                <a href={courses[0].url} class="hover:text-white transition-colors">Quantum Computing</a>
                <a href={courses[1].url} class="hover:text-white transition-colors">AI/ML</a>
              </div>

              <p class="text-gray-500 text-sm">
                Built with ♥ for learners everywhere
              </p>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
