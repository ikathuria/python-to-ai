// Shared behaviour: theme toggle, mobile menu, collapsible lesson contents.
(function () {
	var root = document.documentElement;
	try {
		var saved = localStorage.getItem('p2ai_theme');
		if (saved) root.setAttribute('data-theme', saved);
	} catch (e) { }

	function isDark() {
		var t = root.getAttribute('data-theme');
		if (t) return t === 'dark';
		return window.matchMedia('(prefers-color-scheme: dark)').matches;
	}

	document.addEventListener('DOMContentLoaded', function () {
		var toggle = document.querySelector('.theme-toggle');
		if (toggle) {
			toggle.setAttribute('aria-pressed', isDark());
			toggle.addEventListener('click', function () {
				var next = isDark() ? 'light' : 'dark';
				root.setAttribute('data-theme', next);
				toggle.setAttribute('aria-pressed', next === 'dark');
				try { localStorage.setItem('p2ai_theme', next); } catch (e) { }
			});
		}

		var menuBtn = document.querySelector('.mb-mobile-btn');
		var panel = document.getElementById('mobile-panel');
		if (menuBtn && panel) {
			menuBtn.addEventListener('click', function () {
				var open = panel.classList.toggle('open');
				menuBtn.setAttribute('aria-expanded', open);
			});
		}

		// Close the Lessons dropdown when clicking elsewhere or pressing Escape.
		var dd = document.querySelector('.menubar details');
		if (dd) {
			document.addEventListener('click', function (e) { if (!dd.contains(e.target)) dd.removeAttribute('open'); });
			document.addEventListener('keydown', function (e) { if (e.key === 'Escape') dd.removeAttribute('open'); });
		}

		// Give every form control an accessible name (demo widgets often sit next to, not inside, their label).
		function nameControls() { document.querySelectorAll('main input, main select, main textarea').forEach(function (el) {
			if (el.type === 'hidden' || el.labels && el.labels.length || el.getAttribute('aria-label') || el.getAttribute('aria-labelledby')) return;
			var prev = el.previousElementSibling, text = '';
			while (prev && !text) { text = prev.textContent.trim(); prev = prev.previousElementSibling; }
			if (!text && el.parentElement) {
				var lbl = el.parentElement.querySelector('label, span, p');
				text = lbl ? lbl.textContent.trim() : '';
			}
			if (!text) text = el.placeholder || (el.id || el.name || 'input').replace(/[-_]/g, ' ');
			el.setAttribute('aria-label', text.replace(/\s+/g, ' ').slice(0, 80));
		}); }
		nameControls();
		var main = document.querySelector('main'), pending;
		if (main) new MutationObserver(function () { clearTimeout(pending); pending = setTimeout(nameControls, 100); }).observe(main, { childList: true, subtree: true });

		// Lesson contents: collapsible on phones.
		var aside = document.querySelector('aside');
		var box = aside && aside.querySelector('.sticky');
		if (box && box.querySelector('nav')) {
			var btn = document.createElement('button');
			btn.className = 'toc-toggle';
			btn.type = 'button';
			btn.setAttribute('aria-expanded', 'false');
			btn.innerHTML = '<span>Contents</span><span aria-hidden="true">▾</span>';
			box.insertBefore(btn, box.firstChild);
			aside.classList.add('toc-collapsed');
			btn.addEventListener('click', function () {
				var collapsed = aside.classList.toggle('toc-collapsed');
				btn.setAttribute('aria-expanded', !collapsed);
			});
			box.querySelectorAll('nav a').forEach(function (a) {
				a.addEventListener('click', function () {
					if (window.innerWidth < 768) { aside.classList.add('toc-collapsed'); btn.setAttribute('aria-expanded', 'false'); }
				});
			});
		}
	});
})();
