function initQuiz(questions) {
  const quiz = document.getElementById('quiz');
  if (!quiz) return;

  quiz.innerHTML = '';
  questions.forEach((item, index) => {
    const div = document.createElement('div');
    div.className = 'quiz-question';
    let html = `<h3>${index + 1}. ${item.q}</h3>`;
    item.options.forEach((opt, i) => {
      html += `<label><input type="radio" name="q${index}" value="${i}"> ${opt}</label>`;
    });
    html += `<div class="explanation" id="exp${index}"></div>`;
    div.innerHTML = html;
    quiz.appendChild(div);
  });

  window.submitQuiz = function() {
    let score = 0;
    let answered = 0;
    questions.forEach((item, index) => {
      const selected = document.querySelector(`input[name="q${index}"]:checked`);
      if (selected) {
        answered++;
        if (Number(selected.value) === item.answer) score++;
      }
    });
    const percent = Math.round(score / questions.length * 100);
    const result = document.getElementById('result');
    result.innerHTML = `Kết quả: ${score}/${questions.length} câu đúng (${percent}%). Đã trả lời ${answered}/${questions.length} câu.`;
  };

  window.showReview = function() {
    questions.forEach((item, index) => {
      const exp = document.getElementById(`exp${index}`);
      const selected = document.querySelector(`input[name="q${index}"]:checked`);
      let status = '<span class="wrong">Chưa chọn đáp án.</span>';
      if (selected && Number(selected.value) === item.answer) status = '<span class="correct">Đúng.</span>';
      if (selected && Number(selected.value) !== item.answer) status = '<span class="wrong">Sai.</span>';
      exp.style.display = 'block';
      exp.innerHTML = `${status}<br><strong>Đáp án đúng:</strong> ${item.options[item.answer]}<br><strong>Giải thích:</strong> ${item.explanation}`;
    });
  };

  window.resetQuiz = function() {
    questions.forEach((item, index) => {
      const selected = document.querySelector(`input[name="q${index}"]:checked`);
      if (selected) selected.checked = false;
      const exp = document.getElementById(`exp${index}`);
      exp.style.display = 'none';
      exp.innerHTML = '';
    });
    document.getElementById('result').innerHTML = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
}
