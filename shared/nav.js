(function() {
  var topics = [
    { id: 'home',       label: 'Home',              icon: '🏠', href: 'index.html' },
    { id: 'clustering', label: 'Clustering',         icon: '🔮', href: 'clustering.html' },
    { id: 'bins',       label: 'Bins & Histograms',  icon: '📊', href: 'bins.html' },
    { id: 'failure',    label: 'Failure Detection',  icon: '🔥', href: 'failure-detection.html' },
    { id: 'logdriven',  label: 'Log-Driven Dev',     icon: '📋', href: 'log-driven.html' },
    { id: 'bimodal',    label: 'Bimodal Latency',    icon: '📡', href: 'bimodal.html' },
    { id: 'regression', label: 'Regression',         icon: '📈', href: 'regression.html' },
    { id: 'spikes',     label: 'Spikes & Anomalies', icon: '⚡', href: 'spikes.html' },
  ];

  var currentPath = window.location.pathname.split('/').pop() || 'index.html';
  if (currentPath === '' || currentPath === '/') currentPath = 'index.html';

  var navEl = document.getElementById('site-nav');
  if (!navEl) return;

  var isDark = document.documentElement.classList.contains('dark');

  var linksHTML = topics.map(function(t) {
    var isActive = (currentPath === t.href) ? ' active' : '';
    return '<a class="nav-link' + isActive + '" href="' + t.href + '" title="' + t.label + '">' +
      t.icon + ' <span class="nav-label">' + t.label + '</span>' +
    '</a>';
  }).join('');

  navEl.innerHTML =
    '<nav class="site-nav">' +
      '<div class="nav-links">' +
        linksHTML +
        '<button class="nav-toggle" id="nav-theme-toggle" title="Toggle dark/light mode">' +
          (isDark ? '🌙' : '☀️') +
        '</button>' +
      '</div>' +
    '</nav>';

  var toggleBtn = document.getElementById('nav-theme-toggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', function() {
      if (typeof window.toggleTheme === 'function') {
        window.toggleTheme();
      } else {
        var dark = document.documentElement.classList.contains('dark');
        document.documentElement.classList.toggle('dark', !dark);
        localStorage.setItem('de-theme', dark ? 'light' : 'dark');
      }
      var nowDark = document.documentElement.classList.contains('dark');
      toggleBtn.textContent = nowDark ? '🌙' : '☀️';
    });
  }
})();
