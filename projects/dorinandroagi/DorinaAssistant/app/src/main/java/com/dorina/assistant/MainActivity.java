package com.dorina.assistant;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    private static final int SPEECH_REQUEST_CODE = 101;
    private static final String BRIDGE_SERVER_URL = "http://192.168.1.235:8080/inference";

    private TextView tvChatLog;
    private EditText etCommand;
    private Button btnSend;
    private ImageButton btnMic;
    private ScrollView scrollView;
    private AndroidDeviceActions actions;
    private OkHttpClient httpClient;
    private Handler mainHandler;
    private TextToSpeech tts;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvChatLog = findViewById(R.id.tvChatLog);
        etCommand = findViewById(R.id.etCommand);
        btnSend = findViewById(R.id.btnSend);
        btnMic = findViewById(R.id.btnMic);
        scrollView = findViewById(R.id.scrollView);

        actions = new AndroidDeviceActions(this);
        httpClient = new OkHttpClient();
        mainHandler = new Handler(Looper.getMainLooper());

        // Initialize Android Text-To-Speech (TTS)
        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    tts.setLanguage(new Locale("tr", "TR"));
                    tts.setPitch(1.05f);
                    tts.setSpeechRate(1.0f);
                }
            }
        });

        btnSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String userMsg = etCommand.getText().toString().trim();
                if (!userMsg.isEmpty()) {
                    appendChat("Sen: " + userMsg);
                    etCommand.setText("");
                    processUserMessage(userMsg);
                }
            }
        });

        // Speech-To-Text (STT) Voice Input
        btnMic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startVoiceRecognition();
            }
        });
    }

    private void startVoiceRecognition() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "tr-TR");
        intent.putExtra(RecognizerIntent.EXTRA_PROMPT, "Dorina seni dinliyor...");
        try {
            startActivityForResult(intent, SPEECH_REQUEST_CODE);
        } catch (Exception e) {
            appendChat("Dorina: Ses tanıma başlatılamadı.");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == SPEECH_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            ArrayList<String> results = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
            if (results != null && !results.isEmpty()) {
                String spokenText = results.get(0);
                appendChat("Sen (Sesli): " + spokenText);
                processUserMessage(spokenText);
            }
        }
    }

    private void appendChat(final String text) {
        mainHandler.post(new Runnable() {
            @Override
            public void run() {
                tvChatLog.append(text + "\n\n");
                scrollView.post(new Runnable() {
                    @Override
                    public void run() {
                        scrollView.fullScroll(ScrollView.FOCUS_DOWN);
                    }
                });
            }
        });
    }

    private void speakReply(String text) {
        if (tts != null) {
            // Strip emojis before reading
            String cleanText = text.replaceAll("[\\p{So}\\p{Cn}]", "").trim();
            tts.speak(cleanText, TextToSpeech.QUEUE_FLUSH, null, "dorina_reply");
        }
    }

    private void processUserMessage(String query) {
        // Send to Dorina-V4 bridge on PC first; fallback to offline Turkish fuzzy engine
        JSONObject jsonPayload = new JSONObject();
        try {
            jsonPayload.put("query", query);
        } catch (Exception ignored) {}

        RequestBody body = RequestBody.create(
                jsonPayload.toString(),
                MediaType.parse("application/json; charset=utf-8")
        );

        Request request = new Request.Builder()
                .url(BRIDGE_SERVER_URL)
                .post(body)
                .build();

        httpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                mainHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        handleLocalOfflineIntent(query);
                    }
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    try {
                        String respStr = response.body().string();
                        JSONObject res = new JSONObject(respStr);
                        String tool = res.optString("tool", "");
                        JSONObject args = res.optJSONObject("args");
                        if (args == null) args = new JSONObject();

                        if ("none".equals(tool)) {
                            mainHandler.post(new Runnable() {
                                @Override
                                public void run() {
                                    handleLocalOfflineIntent(query);
                                }
                            });
                            return;
                        }

                        final String result = actions.executeAction(tool, args);
                        mainHandler.post(new Runnable() {
                            @Override
                            public void run() {
                                appendChat("Dorina: " + result);
                                speakReply(result);
                            }
                        });
                    } catch (Exception e) {
                        mainHandler.post(new Runnable() {
                            @Override
                            public void run() {
                                handleLocalOfflineIntent(query);
                            }
                        });
                    }
                } else {
                    mainHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            handleLocalOfflineIntent(query);
                        }
                    });
                }
            }
        });
    }

    /**
     * Highly resilient Turkish offline natural language matcher.
     * Handles typing mistakes: c/ç, s/ş, g/ğ, i/ı, u/ü, o/ö, casing, numbers.
     */
    private void handleLocalOfflineIntent(String text) {
        String norm = TurkishNormalizer.normalize(text);
        String result = "";

        // 1. Brightness / Parlaklık
        if (norm.contains("parlak") || norm.contains("isik") || norm.contains("ekran")) {
            int level = 100;
            Matcher m = Pattern.compile("(\\d{1,3})").matcher(norm);
            if (m.find()) {
                level = Integer.parseInt(m.group(1));
            } else if (norm.contains("kis") || norm.contains("dusur") || norm.contains("azalt") || norm.contains("en az")) {
                level = 15;
            } else if (norm.contains("orta") || norm.contains("yari")) {
                level = 50;
            } else if (norm.contains("full") || norm.contains("kokle") || norm.contains("en yuksek") || norm.contains("ac")) {
                level = 100;
            }
            result = actions.setBrightness(level);
        }
        // 2. Battery / Şarj
        else if (norm.contains("sarj") || norm.contains("batarya") || norm.contains("pil")) {
            result = actions.getBatteryStatus();
        }
        // 3. Volume / Ses
        else if (norm.contains("ses") || norm.contains("volum") || norm.contains("hoparlor")) {
            int vol = 80;
            Matcher m = Pattern.compile("(\\d{1,3})").matcher(norm);
            if (m.find()) {
                vol = Integer.parseInt(m.group(1));
            } else if (norm.contains("sessiz") || norm.contains("kapat") || norm.contains("mute") || norm.contains("kis")) {
                vol = 0;
            } else if (norm.contains("full") || norm.contains("kokle") || norm.contains("en cok")) {
                vol = 100;
            }
            result = actions.setVolume(vol);
        }
        // 4. Flashlight / El Feneri
        else if (norm.contains("fener") || norm.contains("flas") || norm.contains("mesale")) {
            boolean on = !norm.contains("kapat") && !norm.contains("sondur") && !norm.contains("kapa");
            result = actions.toggleFlashlight(on);
        }
        // 5. Alarm
        else if (norm.contains("alarm") || norm.contains("uyandir") || norm.contains("saat")) {
            String time = "08:00";
            Matcher m = Pattern.compile("(\\d{1,2})[:.](\\d{2})").matcher(norm);
            if (m.find()) {
                time = m.group(1) + ":" + m.group(2);
            } else {
                Matcher m2 = Pattern.compile("(\\d{1,2})").matcher(norm);
                if (m2.find()) {
                    time = String.format("%02d:00", Integer.parseInt(m2.group(1)));
                }
            }
            result = actions.setAlarm(time, "Alarm");
        }
        // 6. Open App / Uygulama Aç
        else if (norm.contains("ac") || norm.contains("baslat") || norm.contains("gir") || norm.contains("calistir")) {
            String appName = "Uygulama";
            if (norm.contains("tiktok")) appName = "TikTok";
            else if (norm.contains("youtube") || norm.contains("yutub")) appName = "YouTube";
            else if (norm.contains("whatsapp") || norm.contains("vatsap")) appName = "WhatsApp";
            else if (norm.contains("instagram") || norm.contains("insta")) appName = "Instagram";
            else if (norm.contains("kamera") || norm.contains("camera")) appName = "Kamera";
            else if (norm.contains("galeri") || norm.contains("fotograf")) appName = "Galeri";
            else if (norm.contains("trendyol")) appName = "Trendyol";
            else if (norm.contains("hepsiburada")) appName = "Hepsiburada";
            else if (norm.contains("notlar") || norm.contains("not")) appName = "Notlar";
            else if (norm.contains("twitter") || norm.contains(" x ")) appName = "Twitter";
            result = actions.openApp(appName);
        } else {
            result = "Talha, komutunu tam anlayamadım. Parlaklık, ses, alarm, fener veya uygulama açma isteğini belirtebilirsin!";
        }

        appendChat("Dorina: " + result);
        speakReply(result);
    }

    @Override
    protected void onDestroy() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        super.onDestroy();
    }
}
